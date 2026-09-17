import os
import argparse


def find_continuous_chars(string, min_length):
    left = right = 0
    dict_ele = {}
    while right < len(string):
        if right > left and string[right] != string[right-1]:
            if right - left >= min_length:
                ele = string[left:right]
                key = str(left) + '-' + str(right)
                dict_ele[key] = ele
            left = right
        right += 1
    if right - left >= min_length:
        pass
    return dict_ele
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="min length for secondary structure")
    parser.add_argument("-f", "--file_folder", required=True, type=str, help="secondary structure sequence grad cam score file path")
    parser.add_argument("-l", "--min_len", required=True, type=int, help="min length for secondary structure sequence")
    parser.add_argument("-r", "--res_file", required=True, type=str, help="result file")
    Args = parser.parse_args()  
    
    path = os.path.abspath(Args.file_folder)
    with open(Args.res_file,'w') as w:
      line = 'ID' + ',' + 'SS' + ',' + 'Weight' + ',' + 'SS_sequence_len' + ',' + 'SS_sequence' + ',' + 'File_name(protein_ID)' + ',' + 'SS_ID' + '\n'
      w.write(line)
      
      count = 0
      for file_id in os.listdir(path):
        ff = path + '/' + file_id
        f = open(ff)
        file_name = '.'.join(file_id.split('.')[:-1])
        dict_score = {}
        dict_ss = {}
        ss_list = []
        next(f)
        for i in f:
            id = i.strip().split(',')[0]
            ss = i.strip().split(',')[1]
            score = i.strip().split(',')[2]
            dict_score[id] = score
            dict_ss[id] = ss
            ss_list.append(ss)
        
        ss_str = ''.join(ss_list)
        min_len = int(Args.min_len)
        position = find_continuous_chars(ss_str,min_len)
        ss_id = 0
        
        for key in position:
            ss_id += 1
            start = int(key.split('-')[0])
            end = int(key.split('-')[1])
            if 'H' in position[key]:
                for id in range(start,end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
            elif 'G' in position[key]:
                for id in range(start,end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
            elif 'I' in position[key]:
                for id in range(start, end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
            elif 'E' in position[key]:
                for id in range(start, end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
            elif 'B' in position[key]:
                for id in range(start, end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
            elif 'T' in position[key]:
                for id in range(start, end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
            elif 'S' in position[key]:
                for id in range(start, end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
            elif 'C' in position[key]:
                for id in range(start, end):
                    count += 1
                    key_id = int(id) + 1
                    length = len(position[key])
                    line = str(count) + ',' + dict_ss[str(key_id)] + ',' + dict_score[str(key_id)] + ',' + str(length) + ',' + position[key] + ',' + file_name + ',' + str(ss_id) + '\n'
                    w.write(line)
    w.close()
    
