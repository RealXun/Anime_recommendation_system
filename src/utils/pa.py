import os
import sys
main_folder = os.path.dirname(__file__)
print(main_folder)

main2 = sys.path[0]
print(main2)


create_df(unsupervised_user_based_recommender("Death Note"),"All","All",20)