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
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("-f", "--file_folder", required=True, type=str, help="secondary structure sequence grad cam score file path")
    parser.add_argument("-l", "--min_len", required=True, type=int, help="min length for secondary structure sequence")
    parser.add_argument("-r", "--res_file", required=True, type=str, help="result file")
    Args = parser.parse_args()  
    
    path = os.path.abspath(Args.file_folder)
    
    for file_id in os.listdir(path):
      ff = path + '/' + file_id
      f = open(ff)  
      dict_score = {}
      ss_list = []
      next(f)
      for i in f:
          id = i.strip().split(',')[0]
          ss = i.strip().split(',')[1]
          score = i.strip().split(',')[2]
          dict_score[id] = score
          ss_list.append(ss)
      
      ss_str = ''.join(ss_list)
      min_len = int(Args.min_len)
      position = find_continuous_chars(ss_str,min_len)
      
      H_score = []
      G_score = []
      I_score = []
      E_score = []
      B_score = []
      T_score = []
      S_score = []
      C_score = []
      
      for key in position:
          start = int(key.split('-')[0])
          end = int(key.split('-')[1])
          if 'H' in position[key]:
              for id in range(start,end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  H_score.append(float(mean_val))
          elif 'G' in position[key]:
              for id in range(start,end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  G_score.append(float(mean_val))
          elif 'I' in position[key]:
              for id in range(start, end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  I_score.append(float(mean_val))
          elif 'E' in position[key]:
              for id in range(start, end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  E_score.append(float(mean_val))
          elif 'B' in position[key]:
              for id in range(start, end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  B_score.append(float(mean_val))
          elif 'T' in position[key]:
              for id in range(start, end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  T_score.append(float(mean_val))
          elif 'S' in position[key]:
              for id in range(start, end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  S_score.append(float(mean_val))
          elif 'C' in position[key]:
              for id in range(start, end):
                  key_id = int(id) + 1
                  if end - start == 1:
                    mean_val = float(dict_score[str(key_id)]) / (end - start)
                  else:
                    mean_val = float(dict_score[str(key_id)]) / (end - start - 1)
                  C_score.append(float(mean_val))
    
    if len(H_score) == 0:
      H_average = 0
    else:
      H_average = sum(H_score) / len(H_score)
      
    if len(G_score) == 0:
      G_average = 0
    else:
      G_average = sum(G_score) / len(G_score)
      
    if len(I_score) == 0:
      I_average = 0
    else:
      I_average = sum(I_score) / len(I_score)
    
    if len(E_score) == 0:
      E_average = 0
    else:
      E_average = sum(E_score) / len(E_score)
    
    if len(B_score) == 0:
      B_average = 0
    else:
      B_average = sum(B_score) / len(B_score)
    
    if len(T_score) == 0:
      T_average = 0
    else:
      T_average = sum(T_score) / len(T_score)
      
    if len(S_score) == 0:
      S_average = 0
    else:
      S_average = sum(S_score) / len(S_score)
      
    if len(C_score) == 0:
      C_average = 0
    else:
      C_average = sum(C_score) / len(C_score)
     
    print(H_average)
    
    with open(Args.res_file,'w') as w:
      line = 'H' + ',' + str(H_average) + '\n'
      line = line + 'G' + ',' + str(G_average) + '\n'
      line = line + 'I' + ',' + str(I_average) + '\n'
      line = line + 'E' + ',' + str(E_average) + '\n'
      line = line + 'B' + ',' + str(B_average) + '\n'
      line = line + 'T' + ',' + str(T_average) + '\n'
      line = line + 'S' + ',' + str(S_average) + '\n'
      line = line + 'C' + ',' + str(C_average) + '\n'
      w.write(line)
    w.close()