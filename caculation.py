import re
from bs4 import BeautifulSoup
import os
import csv
# csv_file_path = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta/results/resnet/2025_02_19_05_31_30/cpu_resnet.csv"
import argparse
import yaml

def calculate_indentation(file_path):
    """
    计算指定 .py 文件中每一行的缩进空格数，并根据缩进判断函数体范围。

    :param file_path: 文件路径
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        all_def = {}
        def_stack = []  # 栈，用于追踪嵌套的def函数

        for line_number, line in enumerate(lines, start=1):
            # 去掉行尾的空格或换行符
            stripped_line = line.rstrip()

            # 如果行不为空（避免空行干扰缩进判断）
            if stripped_line:
                # 计算前导空格数
                leading_spaces = len(line) - len(line.lstrip(' '))

                # 判断是否是函数定义
                if stripped_line.lstrip().startswith('def') and len(def_stack) == 0:
                    # 提取函数名
                    func_name = stripped_line.lstrip()[4:].split('(')[0].strip()
                    def_stack.append({'indent': leading_spaces, 'start_line': line_number, 'name': func_name})

                elif stripped_line.lstrip().startswith('def'):
                    if def_stack:
                        current_def = def_stack[-1]
                        if leading_spaces == current_def['indent']:
                            all_def[current_def['name']] = [current_def['start_line'], line_number - 1]
                            def_stack.pop()
                            func_name = stripped_line.lstrip()[4:].split('(')[0].strip()
                            def_stack.append({'indent': leading_spaces, 'start_line': line_number, 'name': func_name})

                # 判断是否是return语句
                elif stripped_line.lstrip().startswith('return'):
                    if def_stack:
                        current_def = def_stack[-1]
                        # 判断是否为当前函数的return
                        if leading_spaces <= current_def['indent']:
                            all_def[current_def['name']] = [current_def['start_line'], line_number]
                            def_stack.pop()
                        elif leading_spaces == current_def['indent'] + 4:
                            all_def[current_def['name']] = [current_def['start_line'], line_number]
                            def_stack.pop()

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径！")
    except Exception as e:
        print(f"发生错误：{e}")
    return all_def


def parse_coverage_report(html_content):
    """
    从 coverage index.html 中提取出 {子页面路径: py文件路径} 的映射关系
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    #print('soup',soup)
    report_data = {}
    file_rows = soup.find_all('tr', class_='region')

    
    for row in file_rows:
        file_link_td = row.find('td', class_='name left')
        if not file_link_td:
            continue

        file_link = file_link_td.find('a')
        if file_link:
            file_key = file_link.get('href')  # 如 'example_py.html'
            # 将链接文字视为 .py 文件路径
            py_file_path = file_link.get_text()  # 如 'example.py'
            report_data[file_key] = py_file_path
    
    return report_data




# parser = argparse.ArgumentParser()
# parser.add_argument("csv_file_path", type=str)
# args = parser.parse_args()


scenarios = [
    {
        'index_html': '/home/cvgroup/myz/czx/semtest-gitee/modelmeta/htmlcov/index.html',
        'htmlcov_dir': '/home/cvgroup/myz/czx/semtest-gitee/modelmeta/htmlcov/'
    }
]


final_result = {}

for scenario in scenarios:
    # 读取当前测试用例的 index.html
    with open(scenario['index_html'], 'r', encoding='utf-8') as f:
        html_content = f.read()
    coverage_data = parse_coverage_report(html_content)
    coverage_result = {}

    for html_file_name, py_file_name in coverage_data.items():
        def_file_path = py_file_name  # 真实的 .py 文件
        all_def = calculate_indentation(def_file_path)
        
        total_functions = len(all_def)
        covered_functions = 0

        # 得到对应的 HTML 报告文件的完整路径（如 /xxx/htmlcov/abc_py.html）
        full_html_path = scenario['htmlcov_dir'] + html_file_name

        # 解析覆盖率详情
        try:
            with open(full_html_path, 'r', encoding='utf-8') as file:
                detail_html = file.read()

            soup_detail = BeautifulSoup(detail_html, 'lxml')
            main_content = soup_detail.find('main', id='source')
            if not main_content:
                # 若 html 结构意外变化，跳过
                coverage_result[def_file_path] = [total_functions, 0]
                continue

            lines = main_content.find_all('p')

            # 遍历每个函数，判断是否覆盖
            for func_name, (start_line, end_line) in all_def.items():
                for code_line_index in range(start_line + 1, end_line):
                    # 检查该行是否标记为 class="run" 并且不是装饰器行
                    if ('class="run"' in str(lines[code_line_index]) 
                        and not lines[code_line_index].text.strip().startswith('@')):
                        covered_functions += 1
                        break

        except FileNotFoundError:
            print(f"HTML 文件 {full_html_path} 未找到，请检查路径！")
        except Exception as e:
            print(f"分析 {full_html_path} 时发生错误：{e}")

        coverage_result[def_file_path] = [total_functions, covered_functions]
    for k, v in coverage_result.items():
        if k not in final_result:
            final_result[k] = [0, 0]
        # 将同名文件的统计结果相加
        final_result[k][0] += v[0]  # 累加总函数数
        final_result[k][1] += v[1]  # 累加覆盖函数数

  
total_func_sum_all = 0
covered_func_sum_all = 0

for _, (total_func_sum, covered_func_sum) in final_result.items():
    total_func_sum_all += total_func_sum
    covered_func_sum_all += covered_func_sum

if total_func_sum_all == 0:
    print("没有统计到任何函数，总覆盖率无法计算。")
else:
    overall_coverage = covered_func_sum_all / total_func_sum_all

    print(f"===> 覆盖率: {overall_coverage:.6%}")  # 输出成百分比格式




basic_dir = "/home/cvgroup/myz/czx/semtest-gitee/modelmeta/"
config_path_main = basic_dir + '/configs/mian.yaml'
with open(config_path_main.lower(), 'r', encoding="utf-8") as f:
    main_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
main_config = main_config_yaml['config']
csv_file_path = main_config["csv_path"]

file_exists = os.path.isfile(csv_file_path)

# 打开CSV文件，以读取模式和追加模式结合打开
with open(csv_file_path, mode='r+', newline='') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)
    
    # 如果文件已经有表头，直接插入
    if not file_exists or len(rows[0]) < 24:  # 如果文件没有表头或者L列不存在
        rows[0].append("coverage")  # 在表头第L列插入coverage
    
    # 写入第二行第L列数据
    if len(rows) > 1:
        rows[1].append(f"{overall_coverage:.6%}")  # 第二行插入overall_coverage值
    else:
        rows.append([f"{overall_coverage:.6%}"])  # 如果第二行不存在，则添加
    
    # 将修改后的数据重新写入文件
    csvfile.seek(0)  # 移动文件指针到开头
    writer = csv.writer(csvfile)
    writer.writerows(rows)