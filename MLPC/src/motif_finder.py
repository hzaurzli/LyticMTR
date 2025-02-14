import os
import argparse


def main(file_path,output_path,cut_off, upper, lower):
    with open(output_path,'w') as w:
      line = 'Motif ID' + ',' + 'Motif sequence' + ',' + 'Length' + ',' + 'Protein ID' + ',' + 'Start' + ',' + 'End' + '\n'
      w.write(line)
      for p in os.listdir(file_path):
          file_name = file_path + '/' + p
          print(p)
          proteinFile = os.path.basename(file_name)
          proteinID = '.'.join(proteinFile.split('.')[:-1])
    
          f = open(file_name)
          dict_ele = {}
          next(f)
          for i in f:
              index = int(i.strip().split(',')[0])
              ele = i.strip().split(',')[1]
              dict_ele[index] = ele
          f.close()
          max_key = max(dict_ele, key=lambda x: x)
      
          f = open(file_name)
          dict_score = {}
          next(f)
          for i in f:
              index = int(i.strip().split(',')[0])
              score = i.strip().split(',')[2]
              dict_score[index] = float(score)
          f.close()
      
          count = 0
          for num in range(lower,upper + 1): # 控制 motif 长度
              f = open(file_name)
              next(f)
              for i in f:
                  index = int(i.strip().split(',')[0])
                  ele = i.strip().split(',')[1]
                  score = i.strip().split(',')[2]
                  if dict_score[index] >= cut_off:
                      if int(index) + int(num) <= max_key:
                          if dict_score[int(index) + int(num)] >= cut_off:
                              list_ele = []
                              list_score = []
                              for key in range(index,int(index) + int(num) + 1):
                                  if dict_score[key] <= cut_off:
                                      list_score.append(dict_score[key])
                                      list_ele.append('X')
                                  else:
                                      list_score.append(dict_score[key])
                                      list_ele.append(dict_ele[key])
                              if sum(s > cut_off for s in list_score) > len(list_score)/2:
                                  motif = ''.join(list_ele)
                                  count += 1
                                  line = 'Motif_' + str(count) + ',' + motif + ',' + str(len(motif)) + ',' + proteinID + ',' + str(index) + ',' + str(int(index) + int(num)) + '\n'
                                  w.write(line)
              f.close()
    w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS motif finder")
    parser.add_argument("-i", "--input", required=True, type=str, help="Grad-CAM position weight file (.csv) path")
    parser.add_argument("-o", "--output", required=True, type=str, help="SS motif information")
    parser.add_argument("-c", "--cut_off", required=True, type=float, help="Grad-CAM position weight cut off")
    parser.add_argument("-mu", "--motif_length_upper", required=True, type=int, help="Motif length upper")
    parser.add_argument("-ml", "--motif_length_lower", required=True, type=int, help="Motif length lower")
    Args = parser.parse_args()
    
    file_path = os.path.abspath(Args.input)
    output_path = Args.output
    cut_off = Args.cut_off
    upper = Args.motif_length_upper
    lower = Args.motif_length_lower
    
    main(file_path, output_path, cut_off, upper, lower)