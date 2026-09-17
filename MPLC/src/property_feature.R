library(tidyverse)
library(dplyr)
library(Biostrings)
library(Peptides)  
library(optparse)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = FALSE,
              action = "store", help = "input protein fasta! format .fa (.fasta)"
  ),
  make_option(c("-o", "--output"), type = "character", default = FALSE,
              action = "store", help = "output property table! format .txt"
  )
)

opt = parse_args(OptionParser(option_list = option_list, 
                              usage = "property feature!"))


fa <- readAAStringSet(opt$input)

table = data.frame(fa) %>%
  rownames_to_column("name") %>%
  mutate("length" = Peptides::lengthpep(seq = fa)) %>% 
  mutate("molecular_weight" = mw(seq = fa)) %>%
  mutate("instability" = instaIndex(seq = fa)) %>%
  mutate("hydrophobicity" = hydrophobicity(seq = fa)) %>%
  mutate("aliphatic" = aIndex(seq = fa)) %>%     
  mutate("pI" = pI(seq = fa)) %>% 
  mutate("charge" = charge(seq = fa)) %>%
  as_tibble()

table = as.data.frame(table[,c(1,3,4,5,6,7,8,9)])
colnames(table) = c("ID","length","molecular_weight",
                    "instability","hydrophobicity","aliphatic",
                    "pI","charge")


write.table(table,opt$output,
          quote = F,row.names = F,sep = '\t')
