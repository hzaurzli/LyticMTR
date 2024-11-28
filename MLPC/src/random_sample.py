import random
import argparse

def main(dict1,dict2,k):
  random_keys = random.choices(list(dict1.keys()), k=k)
  for random_key in random_keys:
    random_value = dict1[random_key]
    dict2[random_key] = random_value
    
  return dict2


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Lysin finder")
  parser.add_argument("-i", "--input_file", required=True, type=str, help="input feature table (txt,'\t')")
  parser.add_argument("-o", "--output_file", required=True, type=str, help="output feature table (after sample, txt,'\t')")
  parser.add_argument("-k", "--sample_k", required=True, type=int, help="sample number")
  Args = parser.parse_args()
  
  f = open(Args.input_file)
  
  old_dict = {}
  new_dict = {}
  for i in f:
    key = i.strip().split('\t')[0]
    val = '\t'.join(i.strip().split('\t')[1:])
    old_dict[key] = val
  
  new_dict = main(old_dict,new_dict,Args.sample_k)
  
  with open(Args.output_file,'w') as w:
    for key in new_dict:
      line = key + '\t' + new_dict[key] + '\n'
      w.write(line)
  w.close()