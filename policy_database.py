"""
policy_database.py — 事例×変数の制度存在データベース生成

【役割】
  analyze.py の Layer A（formal_policy）を論文テキストから切り離し、
  LLMが歴史的知識に基づいて独立に調査・確定する。

  出力: policy_database.csv
    - Case_Name, Variable, Formal_Policy（yes/no）
    - Law_Names（根拠となる法律・条約・制度名）
    - Law_Years（制定年）
    - Notes（補足説明）

  analyze.py はこのCSVを読み込み、
  formal_policy の値を論文評価の前提として注入する。

【実行方法】
  python3 policy_database.py
  → policy_database.csv が生成される（analyze.py 実行前に1回だけ実施）

【Ollamaなし時のフォールバック】
  FALLBACK_POLICY_DB（スクリプト末尾）をそのまま使用する。
  これは研究者が手動で確認・修正可能な固定値として機能する。
"""

import os, json, re, time, requests, warnings
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_NAME     = "qwen3.5:9b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OUTPUT_CSV     = "./policy_database.csv"

PREDEFINED_CASES = [
    "Ainu (Edo Period - Basho Ukeoi)",
    "Ainu (Meiji Period - Former Aborigine Law)",
    "Maori (New Zealand)",
    "Native American (US)",
    "Aboriginal Australians",
    "Taiwan (Japanese Rule)",
    "Korea (Japanese Rule)",
    "Indonesia (Dutch East Indies)",
    "Bengal (British India)",
    "Ireland",
    "Ryukyu (Okinawa)",
]

VARIABLES = [
    ("land_dispossession",   "Land Dispossession",
     "formal legal mechanisms for taking land from indigenous/colonized peoples"),
    ("labor_exploitation",   "Labor Exploitation",
     "formal systems of forced, coerced, or indentured labor imposed on indigenous/colonized peoples"),
    ("cultural_assimilation","Cultural Assimilation",
     "formal policies banning indigenous languages, religions, or cultural practices, or mandating assimilation"),
    ("political_control",    "Political Control",
     "formal administrative structures eliminating indigenous self-governance and imposing colonial authority"),
    ("economic_extraction",  "Economic Extraction",
     "formal taxation, tribute, or trade monopoly systems extracting wealth from indigenous/colonized peoples"),
]

# ==========================================
# 手動確定データベース（研究者が検証済み）
# Ollama未接続時のフォールバック、かつ
# LLM生成結果のベースライン検証用
# ==========================================
FALLBACK_POLICY_DB = [
    # ── Ainu (Edo Period - Basho Ukeoi) ──
    # L:yes La:yes Cu:no Po:yes Ec:yes
    {"Case_Name": "Ainu (Edo Period - Basho Ukeoi)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "場所請負制（漁場としての実質的領有）; 商場知行制",
     "Law_Years": "1600s-1869",
     "Notes": "松前藩が家臣への知行として漁場（場所）を付与。アイヌの土地利用権を制度的に剥奪した枠組み"},

    {"Case_Name": "Ainu (Edo Period - Basho Ukeoi)", "Variable": "labor_exploitation",
     "Formal_Policy": "yes",
     "Law_Names": "場所請負制（漁場での使役労働）",
     "Law_Years": "1700s-1869",
     "Notes": "請負商人がアイヌを漁業労働に強制動員。賃金未払い・強制移住・過労死を伴う制度的強制労働"},

    {"Case_Name": "Ainu (Edo Period - Basho Ukeoi)", "Variable": "cultural_assimilation",
     "Formal_Policy": "no",
     "Law_Names": "N/A",
     "Law_Years": "N/A",
     "Notes": "「異俗」の温存が基本原則。場所請負制は労働力確保のためアイヌ文化を保存した。文化同化政策は明治期以降"},

    {"Case_Name": "Ainu (Edo Period - Basho Ukeoi)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "商場知行制; 場所請負制; 蝦夷地仮上知令（1799）",
     "Law_Years": "1604-1869",
     "Notes": "松前藩がアイヌとの交易・外交を独占。幕府による蝦夷地直轄化（1799, 1807）でさらに強化"},

    {"Case_Name": "Ainu (Edo Period - Basho Ukeoi)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "場所請負制（不平等な交易体制）; 交易独占令",
     "Law_Years": "1700s-1869",
     "Notes": "アイヌが生産した海産物を低対価で強制買取。不平等交易体制による制度的経済収奪"},

    # ── Ainu (Meiji Period - Former Aborigine Law) ──
    # L:yes La:no Cu:yes Po:yes Ec:yes
    {"Case_Name": "Ainu (Meiji Period - Former Aborigine Law)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "北海道旧土人保護法（1899年法律第27号）; 北海道地所規則（1872）",
     "Law_Years": "1872-1899",
     "Notes": "旧土人保護法は「給与地」制度でアイヌの土地所有を制限。北海道地所規則は和人移民への土地無償払下げを規定"},

    {"Case_Name": "Ainu (Meiji Period - Former Aborigine Law)", "Variable": "labor_exploitation",
     "Formal_Policy": "no",
     "Law_Names": "N/A",
     "Law_Years": "N/A",
     "Notes": "直接的な強制労働法規はなし。場所請負制廃止後、農業への強制転換という間接的な労働変容はあるが制度的強制労働ではない"},

    {"Case_Name": "Ainu (Meiji Period - Former Aborigine Law)", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "北海道旧土人保護法（1899）; 第一種学校規程（1901）",
     "Law_Years": "1899-1997",
     "Notes": "旧土人保護法は農業転換を通じた同化を規定。第一種学校規程でアイヌ向け同化教育を制度化。各教育現場でアイヌ語使用が禁止"},

    {"Case_Name": "Ainu (Meiji Period - Former Aborigine Law)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "戸籍法（1871）による編入; 北海道旧土人保護法; 開拓使設置（1869）",
     "Law_Years": "1869-1997",
     "Notes": "戸籍法でアイヌを「平民」として日本の行政体系に強制包摂。開拓使・北海道庁による直接行政統治。アイヌの自治的組織は解体"},

    {"Case_Name": "Ainu (Meiji Period - Former Aborigine Law)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "北海道旧土人保護法（農業奨励を通じた生業制限）; 漁業法によるアイヌ漁業権剥奪",
     "Law_Years": "1886-",
     "Notes": "アイヌの伝統的漁業・狩猟権を法的に剥奪し従来の生業基盤を収奪。旧土人保護法の「給与地」は不毛地が多く経済的収奪と同義"},

    # ── Maori (New Zealand) ──
    # L:yes La:no Cu:yes Po:yes Ec:yes
    {"Case_Name": "Maori (New Zealand)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "ワイタンギ条約の解釈違い（1840）; ニュージーランド入植地法（1863）; New Zealand Settlements Act 1863",
     "Law_Years": "1840-1910",
     "Notes": "NZ Wars後の土地没収（Raupatu）。条約の土地保障条項を英国政府が無視して入植者への土地払下げを制度化"},

    {"Case_Name": "Maori (New Zealand)", "Variable": "labor_exploitation",
     "Formal_Policy": "no",
     "Law_Names": "N/A",
     "Law_Years": "N/A",
     "Notes": "法的な強制労働制度はなし。土地喪失により労働者化が強制されたが、法的制度として確立していない"},

    {"Case_Name": "Maori (New Zealand)", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "先住民学校法（Native Schools Act 1867）; トフンガ抑圧法（Tohunga Suppression Act 1907）",
     "Law_Years": "1867-1907",
     "Notes": "Native Schools Actはマオリ語禁止の英語教育を制度化。Tohunga法は伝統的指導者・治療師の活動を禁止"},

    {"Case_Name": "Maori (New Zealand)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "ニュージーランド憲法法（New Zealand Constitution Act 1852）",
     "Law_Years": "1852-",
     "Notes": "英国統治体系への包摂。マオリ議席は象徴的地位に留められ実質的な自治は否定された"},

    {"Case_Name": "Maori (New Zealand)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "先住民土地法（Native Land Act 1865）",
     "Law_Years": "1865-",
     "Notes": "個人所有化を強制しマオリの共同地を分解。没収・購入した土地の入植者への払下げによる経済的収奪"},

    # ── Native American (US) ──
    # L:yes La:no Cu:yes Po:yes Ec:yes
    {"Case_Name": "Native American (US)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "インディアン移住法（Indian Removal Act 1830）; 一般割当法/ドーズ法（Dawes Act 1887）",
     "Law_Years": "1830-1934",
     "Notes": "Indian Removal Act で強制移住。Dawes法で部族共同地を個人分配後に余剰地を収用、総計90万平方マイルの土地喪失"},

    {"Case_Name": "Native American (US)", "Variable": "labor_exploitation",
     "Formal_Policy": "no",
     "Law_Names": "N/A",
     "Law_Years": "N/A",
     "Notes": "公式な国家による強制労働制度はなし。カリフォルニア州法（1850）等の州レベルの搾取的法は存在するが連邦レベルの強制労働制度ではない"},

    {"Case_Name": "Native American (US)", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "インディアン文明化基金法（Indian Civilization Act 1819）; インディアン寄宿学校政策（1860s-）; インディアン処罰規則（1883）",
     "Law_Years": "1819-1930s",
     "Notes": "寄宿学校制度で先住民言語・文化を強制的に剥奪。'Kill the Indian, Save the Man'政策"},

    {"Case_Name": "Native American (US)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "マーシャル・トリロジー（1823-1832）; Indian Appropriations Act 1871（条約締結の終了）",
     "Law_Years": "1823-",
     "Notes": "マーシャル・トリロジーで部族を国内従属国として規定。1871年法で部族との条約締結を廃止し連邦政府が直接統制"},

    {"Case_Name": "Native American (US)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "ドーズ法（Dawes Act 1887）による信託財産化と資本収奪",
     "Law_Years": "1887-",
     "Notes": "Dawes法で個人配分後の「余剰地」を連邦政府が収用・売却。信託財産制度による経済的支配の継続"},

    # ── Aboriginal Australians ──
    # L:yes La:yes Cu:yes Po:yes Ec:yes
    {"Case_Name": "Aboriginal Australians", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "Terra Nullius（無主の地）の法理; Crown Lands Acts（各州）",
     "Law_Years": "1788-1992",
     "Notes": "Terra Nulliusにより先住民の土地所有権を法的に否定。Mabo判決（1992）まで継続。Crown Lands Actで入植者への土地払下げ"},

    {"Case_Name": "Aboriginal Australians", "Variable": "labor_exploitation",
     "Formal_Policy": "yes",
     "Law_Names": "アボリジニ保護法（Aboriginal Protection Acts 1869~）; 各州保護局による労働と賃金の統制",
     "Law_Years": "1869-1970s",
     "Notes": "保護委員会が先住民の雇用・賃金・移動を統制。Stolen Wages（保護局による賃金横領・未払い）を制度的に合法化"},

    {"Case_Name": "Aboriginal Australians", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "同化政策（Assimilation policy 1937）; 保護法に基づく子供の隔離（盗まれた世代）",
     "Law_Years": "1869-1970",
     "Notes": "1937年の全国同化政策宣言。保護法に基づき先住民の子どもを家族から強制分離（盗まれた世代）し文化的アイデンティティを剥奪"},

    {"Case_Name": "Aboriginal Australians", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "アボリジニ保護法（居住・移動の制限）; Australian Constitution 1901 Section 51xxvi",
     "Law_Years": "1788-1967",
     "Notes": "1901年憲法はアボリジニを人口から除外。保護委員会が生活・移動・結婚に完全な行政統制を行使"},

    {"Case_Name": "Aboriginal Australians", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "Stolen Wages（保護局による賃金横領・未払い）; Terra Nulliusによる土地収用と牧畜業転用",
     "Law_Years": "1869-1970s",
     "Notes": "保護局が先住民労働者の賃金を管理・横領。Terra Nulliusによる土地収用と牧畜業への転用が生業基盤を破壊"},

    # ── Taiwan (Japanese Rule) ──
    # L:yes La:yes Cu:yes Po:yes Ec:yes
    {"Case_Name": "Taiwan (Japanese Rule)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "台湾地籍規則; 林野調査; 官有林野取締規則",
     "Law_Years": "1895-1945",
     "Notes": "台湾土地調査事業（1898-1905）で申告手続き未了地を国有化。林野調査・官有林野取締規則で先住民・農民の慣習的土地利用権を剥奪"},

    {"Case_Name": "Taiwan (Japanese Rule)", "Variable": "labor_exploitation",
     "Formal_Policy": "yes",
     "Law_Names": "保甲制度（労役の提供）; 理蕃政策下の労働供出",
     "Law_Years": "1895-1945",
     "Notes": "保甲制度で地域住民を集団的に労役に動員。理蕃政策下で先住民族に労働供出を強制"},

    {"Case_Name": "Taiwan (Japanese Rule)", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "同化教育; 国語（日本語）普及政策; 皇民化運動（1937-）",
     "Law_Years": "1895-1945",
     "Notes": "皇民化政策で日本語常用・日本式改名・神社参拝を強制。台湾語・中国語の公的使用を禁止"},

    {"Case_Name": "Taiwan (Japanese Rule)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "六三法（1896）（総督への委任立法権）; 保甲法",
     "Law_Years": "1895-1945",
     "Notes": "六三法で台湾総督に立法・行政・軍事の全権を付与。台湾人の政治参加は実質的に排除"},

    {"Case_Name": "Taiwan (Japanese Rule)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "台湾総督府専売局による専売制（樟脳・アヘン等）",
     "Law_Years": "1895-1945",
     "Notes": "主要産品の専売制度により経済的利益を総督府が独占。台湾を日本の原料・食料供給地として再編"},

    # ── Korea (Japanese Rule) ──
    # L:yes La:yes Cu:yes Po:yes Ec:yes
    {"Case_Name": "Korea (Japanese Rule)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "朝鮮土地調査令（1912）; 東洋拓殖株式会社設立（1908）",
     "Law_Years": "1908-1918",
     "Notes": "土地調査で申告なき慣習的土地を国有化。東洋拓殖会社が大規模に農地を収用し日本人移民に払下げ"},

    {"Case_Name": "Korea (Japanese Rule)", "Variable": "labor_exploitation",
     "Formal_Policy": "yes",
     "Law_Names": "国家総動員法（1938）; 国民徴用令（1939）",
     "Law_Years": "1938-1945",
     "Notes": "国家総動員法・国民徴用令で朝鮮人を日本・サハリン・南洋の炭坑・工場に強制連行。推計70万人以上"},

    {"Case_Name": "Korea (Japanese Rule)", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "皇民化政策; 第三次朝鮮教育令（1938）; 創氏改名（1939）",
     "Law_Years": "1938-1945",
     "Notes": "第三次朝鮮教育令でハングル教育禁止・日本語常用強制。創氏改名で朝鮮人固有の姓名を日本式に変更を強制"},

    {"Case_Name": "Korea (Japanese Rule)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "韓国併合ニ関スル条約（1910）; 治安維持法",
     "Law_Years": "1910-1945",
     "Notes": "朝鮮総督に軍政・行政・立法の全権。治安維持法で独立運動を弾圧。大韓帝国の主権を法的に消滅"},

    {"Case_Name": "Korea (Japanese Rule)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "産米増殖計画（1920）; 東洋拓殖株式会社による収奪",
     "Law_Years": "1910-1945",
     "Notes": "米・綿花等の農産物を日本向けに強制供出。朝鮮人は慢性的食料不足に陥る一方、日本へ輸出"},

    # ── Indonesia (Dutch East Indies) ──
    # L:yes La:yes Cu:no Po:yes Ec:yes
    {"Case_Name": "Indonesia (Dutch East Indies)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "農業法（Agrarische Wet 1870）; Domeinverklaring（土地宣言）",
     "Law_Years": "1870-1942",
     "Notes": "Domeinverklaring（土地宣言）でジャワの「無主地」をすべて国有化。Agrarische Wet で大農園への土地貸与を制度化"},

    {"Case_Name": "Indonesia (Dutch East Indies)", "Variable": "labor_exploitation",
     "Formal_Policy": "yes",
     "Law_Names": "強制栽培制度（Cultuurstelsel 1830）; Poenale Sanctie（刑事制裁 1880）",
     "Law_Years": "1830-1942",
     "Notes": "強制栽培制度で農民は収穫物の1/5または60日分の労働を強制。Poenale Sanctionで脱走労働者を刑事罰で強制帰還"},

    {"Case_Name": "Indonesia (Dutch East Indies)", "Variable": "cultural_assimilation",
     "Formal_Policy": "no",
     "Law_Names": "N/A",
     "Law_Years": "N/A",
     "Notes": "強力な同化政策よりも間接統治（indirecte beheersing）を優先。文化・宗教への介入を方針として回避。英日植民地との根本的差異"},

    {"Case_Name": "Indonesia (Dutch East Indies)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "間接統治体制（Regenciesの利用）; VOC独占特許状（1602）",
     "Law_Years": "1602-1942",
     "Notes": "VOCがバタビアを拠点に交易・軍事・行政を独占。現地王侯（Regents）を間接統治の道具として利用"},

    {"Case_Name": "Indonesia (Dutch East Indies)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "強制栽培制度（Cultuurstelsel 1830）による商品作物の独占",
     "Law_Years": "1830-1942",
     "Notes": "強制栽培制度で砂糖・コーヒー・藍等を強制生産・収用。オランダ国庫に1830-1870年で約8億3000万ギルダーを送金"},

    # ── Bengal (British India) ──
    # L:yes La:yes Cu:yes Po:yes Ec:yes
    {"Case_Name": "Bengal (British India)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "ザミーンダーリー制（Permanent Settlement 1793）",
     "Law_Years": "1793-1947",
     "Notes": "永久地税制定令（Permanent Settlement）でザミーンダール制度を法制化。農民の慣習的土地権を剥奪し地主への収奪構造を確立"},

    {"Case_Name": "Bengal (British India)", "Variable": "labor_exploitation",
     "Formal_Policy": "yes",
     "Law_Names": "労働者契約違反法（Workman's Breach of Contract Act 1859）",
     "Law_Years": "1859-1926",
     "Notes": "プランテーション（インディゴ農場等）における年季奉公制を強制。契約違反を刑事罰とし農民の移動の自由を制限"},

    {"Case_Name": "Bengal (British India)", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "マコーレーの教育覚書（Macaulay's Minute on Indian Education 1835）",
     "Law_Years": "1835-",
     "Notes": "英語教育・西洋的教育を公式化。インドの伝統的教育制度を周縁化する政策"},

    {"Case_Name": "Bengal (British India)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "インド統治法（Government of India Act 1858）",
     "Law_Years": "1858-1947",
     "Notes": "東インド会社解散・英国王室直轄統治を制度化。1765年ディワーニー権以来の行政支配が法的に整備"},

    {"Case_Name": "Bengal (British India)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "東インド会社による貿易独占; 重税（地租）; Diwani rights（1765）",
     "Law_Years": "1757-1947",
     "Notes": "1765年のディワーニー権でベンガルの税収を東インド会社が収得。Dadabhai Naoroji推計で年間1億ポンドの「富の流出」"},

    # ── Ireland ──
    # L:yes La:yes Cu:yes Po:yes Ec:yes
    {"Case_Name": "Ireland", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "アイルランド植民（Plantations of Ireland）; アイルランド土地法; Act for Settlement of Ireland 1652",
     "Law_Years": "1556-1829",
     "Notes": "クロムウェル法でカトリック所有地の大規模没収。アイルランド人地主のConnacht以西への強制移住命令"},

    {"Case_Name": "Ireland", "Variable": "labor_exploitation",
     "Formal_Policy": "yes",
     "Law_Names": "刑罰法（Penal Laws）（カトリック教徒の職業・労働制限）",
     "Law_Years": "1695-1829",
     "Notes": "刑事法下でカトリック系アイルランド人の職業選択を制限。コナクレ制度で農民を土地なし小作人として搾取"},

    {"Case_Name": "Ireland", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "刑罰法（Penal Laws）（カトリック教育およびゲール語の制限）; Statutes of Kilkenny 1366",
     "Law_Years": "1366-1829",
     "Notes": "キルケニー法でアイルランド語・アイルランド習慣を禁止。刑事法でカトリック教育・礼拝を違法化"},

    {"Case_Name": "Ireland", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "連合法（Act of Union 1800）",
     "Law_Years": "1800-1922",
     "Notes": "合同法でアイルランド議会を廃止。カトリックの議会参加を1829年まで禁止"},

    {"Case_Name": "Ireland", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "不在持主制（Absentee landlordism）; 穀物法（Corn Laws）",
     "Law_Years": "1660-1846",
     "Notes": "羊毛法でアイルランドの毛織物産業を壊滅。不在地主の地代収入が英国へ流出"},

    # ── Ryukyu (Okinawa) ──
    # L:yes La:no Cu:yes Po:yes Ec:yes
    {"Case_Name": "Ryukyu (Okinawa)", "Variable": "land_dispossession",
     "Formal_Policy": "yes",
     "Law_Names": "沖縄県土地整理法（1899）",
     "Law_Years": "1899-1903",
     "Notes": "土地整理事業で琉球王府時代の共同地・地割制を廃止。土地の個人所有化を強制し国有地として収用"},

    {"Case_Name": "Ryukyu (Okinawa)", "Variable": "labor_exploitation",
     "Formal_Policy": "no",
     "Law_Names": "N/A",
     "Law_Years": "N/A",
     "Notes": "沖縄県民に特化した強制労働法はなし。本土と同一の法制度下に置かれた。砂糖産業での実質的搾取は存在するが法的制度として確立していない"},

    {"Case_Name": "Ryukyu (Okinawa)", "Variable": "cultural_assimilation",
     "Formal_Policy": "yes",
     "Law_Names": "標準語励行運動（方言札など）; 沖縄県教育令",
     "Law_Years": "1890s-1945",
     "Notes": "方言（沖縄語）使用者に「方言札」を掛けさせる屈辱的制度で日本語使用を強制。戦時中は皇民化で琉球文化を抑圧"},

    {"Case_Name": "Ryukyu (Okinawa)", "Variable": "political_control",
     "Formal_Policy": "yes",
     "Law_Names": "琉球処分（1879）; 府県制の遅延導入",
     "Law_Years": "1879-1920",
     "Notes": "1879年の「琉球処分」で琉球王国を廃し沖縄県を強制設置。府県制の遅延導入により本土より遅く参政権が付与され統治上の差別が存在"},

    {"Case_Name": "Ryukyu (Okinawa)", "Variable": "economic_extraction",
     "Formal_Policy": "yes",
     "Law_Names": "人頭税の旧慣温存（1903年まで）",
     "Law_Years": "1609-1903",
     "Notes": "薩摩藩時代からの人頭税制度を明治後も「旧慣温存」として1903年まで継続。本土より重い税負担"},
]




# ==========================================
# LLMによる補完・検証（Ollama使用時）
# ==========================================
def ask_llm(prompt, timeout=120):
    try:
        r = requests.post(OLLAMA_ENDPOINT,
                          json={"model": MODEL_NAME, "prompt": prompt,
                                "stream": False, "format": "json"},
                          timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response")
    except Exception:
        pass
    return None

def parse_json(raw):
    if not raw: return None
    try: return json.loads(raw)
    except Exception: pass
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None


def verify_and_supplement_with_llm(existing_db):
    """
    LLMでデータベースを補完する。

    【重要な設計方針】
    - "no → yes" の訂正は原則として禁止する
      理由: LLMは「植民地支配に何らかの法的根拠があったはず」という
            確証バイアスで架空の法律名を生成する傾向がある
            （例: "Shinmatsu Hokan Tsusho", "Eindabzijdig Beleid"等は実在しない）
    - "Locked": True の項目は変更不可（理論的に重要な "no"）
    - LLMの役割は "yes" 項目の法律名補完のみ
    - "no → yes" の変更は Locked=False かつ confidence=high のみ許可し、
      さらに架空法律名フィルタを通す
    """
    # 理論的に重要な "no" はロック（変更不可）
    # これらは「制度がない」こと自体が分析上の意味を持つ
    LOCKED_NO_ENTRIES = {
        # 精査版テーブルで確認された全6件の「no」エントリをロック
        ("Ainu (Edo Period - Basho Ukeoi)", "cultural_assimilation"):
            "「異俗」の温存が基本原則。場所請負制は文化保存が前提。文化同化政策の欠如が搾取型類型の核心",
        ("Ainu (Meiji Period - Former Aborigine Law)", "labor_exploitation"):
            "直接的な強制労働法規はなし。農業強制転換は間接的変容であり制度的強制労働ではない",
        ("Maori (New Zealand)", "labor_exploitation"):
            "法的な強制労働制度はなし。土地喪失による労働者化は間接的強制",
        ("Native American (US)", "labor_exploitation"):
            "公式な国家による強制労働制度はなし。カリフォルニア州法等は州レベルであり連邦制度でない",
        ("Indonesia (Dutch East Indies)", "cultural_assimilation"):
            "間接統治（indirecte beheersing）を優先し文化介入を方針として回避。オランダ植民地の根本的特徴",
        ("Ryukyu (Okinawa)", "labor_exploitation"):
            "沖縄県民に特化した強制労働法はなし。本土と同一の法制度下",
    }

    # 架空法律名を検出するためのチェックリスト
    SUSPICIOUS_PATTERNS = [
        r"eindabzijdig",
        r"kyōritsu.*kiseichō",
        r"shinmatsu",
        r"land wars act",
        r"edict on.*governance.*ryukyu",
        r"treaty of shimonoseki.*ryukyu",
        r"treaty of shimonoseki.*labor",
        r"shimonoseki.*1895.*ryukyu",
        r"shimonoseki.*1895.*labor",
        r"governance.*ryukyu.*1609",
        r"forced labor act.*ryukyu",
        r"forced labor act.*okinawa",
        r"edict.*ryukyu.*labor",
        # 一般的な架空パターン（あいまいな制度名）
        r"colonial labor ordinance",
        r"indigenous labor regulation",
    ]

    def is_suspicious_law_name(law_name):
        law_lower = law_name.lower()
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, law_lower):
                return True
        return False

    print("  LLMによるデータベース補完を開始...")
    print("  ※ 「no → yes」の訂正は架空法律名防止のため厳格にフィルタします")
    updated = []

    for item in existing_db:
        case     = item["Case_Name"]
        variable = item["Variable"]
        fp       = item["Formal_Policy"]

        # ロックされた "no" エントリは変更不可
        lock_key = (case, variable)
        if lock_key in LOCKED_NO_ENTRIES and fp == "no":
            item = {**item,
                    "Locked": True,
                    "LLM_Verified": "locked",
                    "Notes": f"{item.get('Notes','')} [LOCKED: {LOCKED_NO_ENTRIES[lock_key]}]"}
            updated.append(item)
            continue

        if fp == "no":
            # ロックされていない "no" のみLLMで確認
            prompt = f"""
/think
You are a colonial history expert. I need to verify whether there was a FORMAL governmental 
policy for the following situation:

Colonial case: "{case}"
Dimension: "{variable}"

STRICT CRITERIA for "yes":
  - Must be a REAL, verifiable law/treaty/policy with a known official name
  - Must be specifically about {variable} (not a general colonial law)
  - You must be CERTAIN this law exists and has the exact name you provide
  - Do NOT invent plausible-sounding names. If uncertain, answer "no"

IMPORTANT: Many colonial situations had no FORMAL policy for specific dimensions.
For example:
  - Ainu Edo period: Cultural assimilation policy did NOT exist (labor supply required cultural preservation)
  - Dutch East Indies: Cultural assimilation policy did NOT exist (indirect rule avoided cultural intervention)
  - These "no" answers are historically correct and should remain "no"

OUTPUT EXACTLY:
{{"formal_policy": "yes" or "no",
  "law_names": "EXACT official name(s) or N/A",
  "law_years": "year(s) or N/A",
  "confidence": "high" or "medium" or "low",
  "notes": "brief factual explanation"}}
"""
            raw = ask_llm(prompt, timeout=60)
            if raw:
                data = parse_json(raw)
                if data:
                    llm_fp   = str(data.get("formal_policy","no")).lower()
                    conf     = str(data.get("confidence","low")).lower()
                    law_name = str(data.get("law_names","N/A"))

                    # 「no → yes」の訂正条件を厳格化:
                    # 1. confidence = "high" のみ（mediumは却下）
                    # 2. 架空法律名フィルタを通過
                    if llm_fp == "yes" and conf == "high":
                        if is_suspicious_law_name(law_name):
                            print(f"    ⚠️  架空法律名を検出し却下: [{case}][{variable}] "
                                  f"'{law_name[:40]}'")
                            item = {**item, "LLM_Verified": "rejected_suspicious",
                                    "Locked": False}
                        else:
                            print(f"    → 補完: [{case}][{variable}] no → yes "
                                  f"('{law_name[:40]}') ※要人間確認")
                            item = {**item,
                                    "Formal_Policy": "yes",
                                    "Law_Names": law_name,
                                    "Law_Years": str(data.get("law_years","N/A")),
                                    "Notes": f"[LLM補完・要確認] {data.get('notes','')}",
                                    "LLM_Verified": "suggested_needs_review",
                                    "Locked": False}
                    else:
                        # LLMも "no" と判断、または confidence が low/medium
                        item = {**item, "LLM_Verified": "confirmed_no", "Locked": False}
                else:
                    item = {**item, "LLM_Verified": "parse_failed", "Locked": False}
            else:
                item = {**item, "LLM_Verified": "no_response", "Locked": False}

        elif fp == "yes":
            # "yes" 項目: 法律名の補完のみ（Formal_Policy は変更しない）
            if not item.get("Law_Names") or item.get("Law_Names") == "N/A":
                # 法律名が未設定の場合のみ補完
                prompt = f"""
/no_think
Provide the specific official law/treaty/policy name for:
Case: "{case}", Dimension: "{variable}"
Only provide names you are CERTAIN exist. Output:
{{"law_names": "exact name(s)", "law_years": "year(s)"}}
"""
                raw = ask_llm(prompt, timeout=45)
                if raw:
                    data = parse_json(raw)
                    if data and data.get("law_names"):
                        item = {**item,
                                "Law_Names": data.get("law_names", item["Law_Names"]),
                                "Law_Years": data.get("law_years", item["Law_Years"]),
                                "LLM_Verified": True, "Locked": False}
            else:
                item = {**item, "LLM_Verified": True, "Locked": False}

        updated.append(item)

    # 補完結果のサマリー
    suggested = [r for r in updated if r.get("LLM_Verified") == "suggested_needs_review"]
    rejected  = [r for r in updated if r.get("LLM_Verified") == "rejected_suspicious"]
    locked    = [r for r in updated if r.get("Locked") is True]

    print(f"\n  LLM補完サマリー:")
    print(f"    ロック済み（変更不可）: {len(locked)}件")
    print(f"    補完提案（要人間確認）: {len(suggested)}件")
    print(f"    架空法律名で却下:       {len(rejected)}件")
    if suggested:
        print(f"  ⚠️  以下の「no→yes」補完は policy_database.csv で手動確認してください:")
        for r in suggested:
            print(f"    [{r['Case_Name']}][{r['Variable']}]: {r['Law_Names'][:50]}")

    return updated


def main():
    print("="*60)
    print("  制度存在データベース生成（policy_database.py）")
    print("="*60)

    # Ollama確認
    llm_available = False
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        llm_available = any(MODEL_NAME.split(":")[0] in m for m in models)
        print(f"\n  Ollama: {'接続OK → LLM検証あり' if llm_available else '未接続'}")
    except Exception:
        print("\n  Ollama: 未接続 → 手動データベースのみ使用")

    print(f"\n  手動確認済みデータ: {len(FALLBACK_POLICY_DB)}件")
    print(f"  事例数: {len(PREDEFINED_CASES)}  変数数: {len(VARIABLES)}")

    db = FALLBACK_POLICY_DB.copy()

    if llm_available:
        db = verify_and_supplement_with_llm(db)

    df = pd.DataFrame(db)

    # 確認用サマリー
    print("\n" + "="*60)
    print("  制度存在サマリー（yes/no の分布）")
    print("="*60)
    print(f"\n  {'事例':<45} {'L':>3} {'La':>3} {'Cu':>3} {'Po':>3} {'Ec':>3}")
    print("  " + "-"*60)

    var_keys = ["land_dispossession","labor_exploitation","cultural_assimilation",
                "political_control","economic_extraction"]

    for case in PREDEFINED_CASES:
        case_rows = df[df["Case_Name"] == case]
        flags = []
        for v in var_keys:
            row = case_rows[case_rows["Variable"] == v]
            if len(row) > 0:
                fp = row.iloc[0]["Formal_Policy"]
                flags.append("Y" if fp=="yes" else "N" if fp=="no" else "?")
            else:
                flags.append("?")
        print(f"  {case:<45} {' '.join(f'{f:>3}' for f in flags)}")

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n  → {OUTPUT_CSV} に保存完了（{len(df)}件）")
    print("""
  【次のステップ】
    1. policy_database.csv を確認・手動修正（必要であれば）
    2. analyze.py を実行（policy_database.csv を自動参照）
""")


if __name__ == "__main__":
    main()
