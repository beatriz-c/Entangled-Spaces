# Entangled-Spaces
                                                                                          Pipeline

Tranfers:

1 - 0datasets_creation_global - usar 0globaldata.csv - retira-se 0globaldata_nd.csv que consiste em todos os dados após remover os duplicados; retira-se 0transfers_nd.csv que reúne todas 
                                                       as transfers após remoção dos duplicados 


2 - 1operativas_analysis_transfers - usar 0transfers_nd.csv - anotar as operativas a eliminar em transfers


3 - 0datasets_creation_global - usar 0transfers_nd.csv - selecionar as operativas que ficam; retirar 0transfers_nd_corr.csv que não tem duplicados e tem a análise da correlação feita 
                                                         (este é o ficheiro final das transferências; retirar 0frauds_transfers_nd_corr.csv que tem apenas as fraudes das transferências 
                                                         após remoção de duplicados e da análise de correlação; retirar 0nonfrauds_transfers_nd_corr.csv que tem apenas as não-fraudes 
                                                         das transferências após remoção de duplicados e da análise de correlação


4 - 2smote+rund_transfers - usar 0transfers_nd_corr.csv - mudar a sampling strategy - faz random undersampling para selecionar aleatoriamente não-fraudes e faz o smote para gerar fraudes 
                                                                                      sintéticas; retirar 2transfers_balanced_smote+rund.csv que contém o dataset balanceado; retirar 
                                                                                      report 2transfers_balanced_smote+rund.html que é o profile do dataset balanceado; retirar 
                                                                                      2frauds_transfers_balanced_smote+rund.csv que contém apenas as fraudes do dataset balanceado 


5 - 2rund_transfers - usar 0transfers_nd_corr.csv - mudar a sampling strategy - faz random undersampling para selecionar aleatoriamente não-fraudes; retirar 2transfers_rund.csv que tem
                                                                                o dataset com menos não-fraudes e continua não balanceado; retirar report 2transfers_rund.html que é o
                                                                                profile do dataset com apenas uma sample das não fraudes e não balanceado 


6 - 2gan_transfers - usar 2transfers_rund.csv - gera fraudes sintéticas através de uma gan; retirar 2transfers_balanced_gan+rund.csv que contém o dataset balanceado; retirar 
                                                2transfers_balanced_gan+rund.csv que é o profile do dataset balanceado; retirar 2frauds_transfers_balanced_gan+rund.csv que tem apenas as
                                                fraudes do dataset balanceado


7 - 3transfers_word2vec_matrix_originalcols - alterar número de bins com base nos profile reports - ver vocabulário - usar 2frauds_transfers_balanced_gan+rund.csv - retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols.w2v que tem o modelo treinado;
                                                                                                                      retirar 3transfers_word2vec_matrix_originalcols_density_matrix.csv 
                                                                                                                      que tem a density matrix; retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols_prob_vector.csv que tem o vetor
                                                                                                                      de probabilidades; retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols_eingenvalues.csv que tem os
                                                                                                                      valores próprios da matriz produto antes do softmax; retirar scores
												                    
														    - usar 2transfers_balanced_smote+rund.csv - retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols.w2v que tem o modelo treinado;
                                                                                                                      retirar 3transfers_word2vec_matrix_originalcols_density_matrix.csv 
                                                                                                                      que tem a density matrix; retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols_prob_vector.csv que tem o vetor
                                                                                                                      de probabilidades; retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols_eingenvalues.csv que tem os
                                                                                                                      valores próprios da matriz produto antes do softmax; retirar scores

                                                                                                                    - usar 2transfers_rund.csv (imbalanced) - retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols.w2v que tem o modelo treinado;
                                                                                                                      retirar 3transfers_word2vec_matrix_originalcols_density_matrix.csv 
                                                                                                                      que tem a density matrix; retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols_prob_vector.csv que tem o vetor
                                                                                                                      de probabilidades; retirar 
                                                                                                                      3transfers_word2vec_matrix_originalcols_eingenvalues.csv que tem os
                                                                                                                      valores próprios da matriz produto antes do softmax; retirar scores

                                                                                                                    - experimentar os 3 datasets sem a coluna da fraude e do month

                                                                                                                    - experimentar sem ordenar as colunas da dataframe para os 3 datasets


8 - 3transfers_word2vec_metrics_originalcols - alterar número de bins com base nos profile reports - ver vocabulário - usar 2frauds_transfers_balanced_gan+rund.csv - retirar scores
												                     - usar 2transfers_balanced_smote+rund.csv - retirar scores
                                                                                                                     - usar 2transfers_rund.csv (imbalanced) - retirar scores

                                                                                                                     
9 - 4graphics_word2vec - usar 3transfers_word2vec_matrix_originalcols.w2v - retirar gráficos
                       - usar 3transfers_word2vec_metrics_originalcols.w2v - retirar gráficos


10 - 5rund_transfers_with_timestamp - usar 0transfers_nd_corr.csv - mudar a sampling strategy - retirar 5transfers_rund.csv que contém o dataset com a coluna timestamp e faz random 
                                                                                                undersampling


11 - 5gan_transfers_with_timestamp - usar retirar 5transfers_rund.csv  - retirar 5transfers_balanced_gan+rund.csv que contém o dataset balanceado com a coluna timestamp 


12 - 5transfers_column_creation - usar 5transfers_balanced_gan+rund.csv - retirar 5transfers_balanced_gan+rund_allc.csv que contém o dataset balanceado com as colunas novas; 
                                                                          retirar 5transfers_balanced_smote+rund_allc.html que contém o profile das novas colunas (até das que não vão ser 
                                                                          usadas para treinar o modelo)


13 - 5transfers_word2vec_matrix_allcolumns ou 5transfers_word2vec_metrics_allcolumns conforme o que tiver melhor performance - alterar número de bins com base nos profile reports -
                                                                                                                               ver vocabulário - 
 falta!---------                                                                                                                                usar 5transfers_balanced_smote+rund_allc.csv
                                                                                                                               - retirar scores 


14 - 6frauds_transfers - usar 0frauds_transfers_nd_corr.csv - retirar 0frauds_transfers_nd_corr.html que é o profile das fraudes originais antes da geração sintética; 
                       - usar 2frauds_transfers_balanced_smote+rund.csv - retirar 2frauds_transfers_balanced_smote+rund.html que é o profile das fraudes originais mais as geradas pelo smote;
                       - usar 2frauds_transfers_balanced_gan+rund.csv - retirar 2frauds_transfers_balanced_gan+rund.html que é o profile das fraudes originais mais as geradas por gans