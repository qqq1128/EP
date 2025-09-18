##生成JSON
import pandas as pd
import re
import os




# 读取prompt.txt
def read_prompt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def eval_math_expressions(text):
    # 支持括号、数字、空格和+-*/，也支持单独数字（整数或小数）
    pattern = r"\{([0-9+\-*/\s\(\)\.]+)\}"
    def replacer(match):
        expr = match.group(1).strip()
        try:
            value = eval(expr)
            if isinstance(value, float):
                value = round(value, 6)
            return str(value)
        except Exception:
            return match.group(0)
    return re.sub(pattern, replacer, text)

prompt_content = read_prompt_file('prompt.txt')

# 读取final_combined_results_9.2.csv
combined_df = pd.read_csv('final_combined_results_(with response).csv', dtype=str)
# 以补零的方式补到6位
combined_df['tract'] = combined_df['tract'].astype(int).apply(lambda x: f"{x:06d}").astype(str)

# 对combined_df列进行排序，表头长度越长越先处理
combined_df = combined_df.reindex(sorted(combined_df.columns, key=len, reverse=True), axis=1)

# 遍历每一行combined_df
fillprompt=[]
for index, row in combined_df.iterrows():
    prompt_content_tmp = prompt_content
    # 仅替换花括号内部的占位符内容
    # 先构建列名按长度排序（已在 DataFrame 层排序），避免较短列名抢先替换
    columns_sorted = list(combined_df.columns)
    # 匹配 { ... } 占位符；只在占位符内部做列名替换
    placeholder_pattern = re.compile(r"\{([^{}]+)\}")

    def replace_inside_braces(m):
        inner = m.group(1)
        original_inner = inner
        for col in columns_sorted:
            #print(f"Processing column: {col}")
            # 使用单词边界/或前后是开始结束/非字母数字分隔，兼容列名含空格。
            # 由于列名可能包含空格，\b 可能不足；采用前后断言：(^|[^A-Za-z0-9_])col([^A-Za-z0-9_]|$)
            escaped = re.escape(col)
            inner = re.sub(rf'(?i)(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])', str(row[col]), inner)
        if inner != original_inner:
            print(f"Replaced in placeholder: {{{original_inner}}} -> {{{inner}}}")
        return '{' + inner + '}'

    prompt_content_tmp = placeholder_pattern.sub(replace_inside_braces, prompt_content_tmp)

    # 计算完全数字/算术表达式的占位符
    prompt_content_tmp = eval_math_expressions(prompt_content_tmp)
    prompt_content_tmp = prompt_content_tmp.replace('\n', '')
    fillprompt.append(prompt_content_tmp)

# 保存结果
with open('fillprompt.jsonl', 'w', encoding='utf-8') as f:
    f.write('\n'.join(fillprompt))




import openai
import json
import csv
import pandas as pd
openai.api_key = ""
file_path = r"C:\Users\qianyi\Desktop\code\empr\FLmiami.xlsx"
df = pd.read_excel(file_path)
preference =df['Bidding preference']
print(preference)
###  Bidding preference interpreter
bidding_ranges = []
for preference in df['Bidding preference']:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in pricing and market analysis. Your task is to interpret market participants' bidding price preferences, described in natural language, into numerical price ranges. In a double auction-based P2P electricity market, the bidding range for all participants is constrained within [Feed-in-tariff (FiT), Time-of-use (ToU)]."},
            {"role": "user", "content": f"Based on the following description {preference}, please generate the price bidding range for this community. Response Format: A range, where the lower and upper bounds are represented as multiples of FiT and ToU, respectively. That is, [xFiT, yToU], where x > 1 and y < 1. Note that in our case, FiT = 0.25 * ToU, and the generated bidding range should satisfy xFiT < yToU. Generate only the range, without explanation."},
        ],
            temperature= 0.4

    )

    # 获取返回的 bidding range 内容
    bidding_range = response.choices[0].message.content
    print(bidding_range)
    # 将结果添加到列表中
    bidding_ranges.append(bidding_range.strip())

# 将所有结果写入 CSV 文件
outcome_df = pd.DataFrame(bidding_ranges, columns=['range'])
# 输出文件路径
output_file = r"C:\Users\qianyi\Desktop\code\empr\outcome.csv"
# 将数据框写入 CSV 文件
outcome_df.to_csv(output_file, index=False)
print(f"Generated bidding ranges have been saved to {output_file}")

###~ Energy access index evaluator
file_path =  r"C:\Users\qianyi\Desktop\code\empr\FLmiami.xlsx" # 如果文件路径有变化，请更新此路径
data = pd.read_excel(file_path)
data_cleaned = pd.read_excel(file_path, header=1)
for column in data_cleaned.columns:
    # 尝试将列数据转换为数值类型（忽略非数值项）
    data_cleaned[column] = pd.to_numeric(data_cleaned[column], errors='coerce')
    # 只处理大于 0 且小于 1 的值
    data_cleaned[column] = data_cleaned[column].apply(lambda x: f"{x * 100:.2f}%" if 0 < x < 1 else x)

import re
import time
EAI =[]
# for index, row in data_cleaned.iterrows():
# for index, row in data_cleaned.iloc[:20].iterrows():
for index, row in data_cleaned.head(10).iterrows():
    information = (f"(1) Demographic Groups: The total population of this area is {row['Total population']} accompanied by a median family income of {row['Median family income']}. The demographic breakdown reveals that {row['% Persons with disabilities']}  of the population are with disabilities, while {row['% People of color']} are people of color. Among the racial groups, {row['%White']} are white, {row['%Black']} are black, {row['%American Indian and Alaska']} are American Indian, and {row['%Asian']} are Asian. Furthermore, the unemployment rate stands at {row['% Unemployed']}, and {row['% Speak English less than very well']} of the population faces language barriers.  {row['% Less than high school']} lack a high school diploma. Additionally, and {row['% Low life expectancy']} of the population have low life expectancy. The economic landscape includes varied monthly electricity costs, where {row['%$50 to $99 Monthly Electricity Costs'] } pay between $50 to $99 for monthly electricity, while {row['%Less than $50 Monthly Electricity Costs'] } pay less than $50, and {row['%$100 to $149 Monthly Electricity Costs'] } pay $100 to $149, {row['%$150 to $199 Monthly Electricity Costs']} pay $150 to $199, {row['%$200 to $249 Monthly Electricity Costs']} pay $200 to $249, and {row['%$250 or more Monthly Electricity Costs']} pay more tham $250. Additionally, {row['%Under 5']} are under the age of five, while {row['%Over 64']} are over the age of 64. (2) Environmental Risks: The proximity to hazardous sites is a key concern, with the community being near Superfund sites, as indicated by {row['Superfund proximity']}. Similarly, {row['RMP facility proximity']} shows the presence of Risk Management Plan (RMP) facilities. Additionally, {row['Hazardous waste proximity']} highlights the community’s exposure to hazardous waste sites."
                f"(3) Infrastructure and Service Accessibility: It is characterized by {row['Traffic proximity']} traffic proximity.  {row['% Private vehicle'] } of the population rely on private vehicles, while {row['%Public transportation'] } depend on public transport systems. Housing affordability is also a key issue, with median real estate taxes paid standing at {row['Median real estate taxes paid']} and median gross rent being {row['Median gross rent']}. Residents face a median value of monthly housing costs, with a median value of {row['Median monthly housing costs']}.  (4) Living Conditions: The housing stock in this area is diverse, with {row['%1, detached Units in Structure']} of the homes being single-family detached units, while {row['%1, attached Units in Structure'] } live in attached housing units. Additionally, {row['%Total 50 or more']} reside in structures with 50 or more units, while {row['%Mobile home Units in Structure']} live in mobile homes. Ownership rates reveal that {row['%Owner occupied housing units']} of the homes are owner-occupied, while {row['%Renter occupied housing units']} are renter-occupied.  {row['% Built before 1960']} of the buildings are older than 1960.")
    print(information)
    # API 调用
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in energy policy and energy market. Your task is to assess the energy poverty level of the American census tracts. We define the Energy Access Index (EAI) as an energy poverty indicator, where 1 represents extreme energy poverty and 10 represents extremely affluent energy access."},
            {"role": "user", "content": f"Here is the energy poverty-related information of a census tract in Miami, Florida. Please analyze all features comprehensively and calculate the EAI for each census tract based on the multidimensional information below. {information}  In summary, energy poverty in this census tract is shaped by Demographic Groups, Environmental Risks, Infrastructure and Service Accessibility and Living conditions. Based on these factors, calculate the EAI. The output should only be a decimal between 1 and 10. Please provide interpretation."}

        ],
        temperature=0.2
    )
    EAI_score = response.choices[0].message.content.strip()
    # EAI_score = float(response.choices[0].message.content.strip())
    print(EAI_score)
    EAI.append(EAI_score)
    # 延迟5秒，避免超出API调用限制
    time.sleep(5)

# 将结果转换为 DataFrame，便于查看和分析
EAI_df = pd.DataFrame(EAI, columns=['EAI Score'])
# 显示计算结果
print(EAI_df)




### Carbon allowance allocator
for index, row in data_cleaned.iterrows():
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
         {"role":"system","content": "You are an expert in carbon emissions and energy policy. Your task is to allocate the initial free carbon allowance for a census tract. The principle is that a census tract with a high EAI and a bidding range  with high risk tolerance should bear a greater carbon reduction responsibility (i.e., have a relatively samll allowance coefficient)."},
          {"role": "user", "content": f"The basic allowance coefficient are as follows: load 0.38; photovoltaic 0.20; microturbine 0.46, which respresnt the amount of free carbon emissions (in kgCO2/kWh) that can be allocated per unit of power. Please adjust the allowance coefficient coefficient for the census tract based on its bidding range and energy acees index result {EAI}."}
          ],
        temperature=0.2
    )
print(response.choices[0].message.content)








