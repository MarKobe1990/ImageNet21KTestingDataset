import os

from treelib import Tree
import pickle
import shutil


def create_tree(source_fn_path):
    word_tree = Tree()
    with open(source_fn_path, 'rb') as f:
        word_tree = pickle.load(f)
    return word_tree


def output_dataset(input_word, word_tree, save_dir):
    input_word_node = word_tree.get_node(input_word)
    img_path_list = input_word_node.data.files_path_list
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for img_path in img_path_list:
        shutil.copyfile(img_path, save_dir + img_path.split('/')[-1])
        # # 打开源图片
        # source_img = open(img_path, "rb")
        # # 写入地址
        # cp_img = open(save_dir + img_path.split('/')[-1], "wb")
        # cp_img.write(source_img.read())
        #
        # source_img.close()
        # cp_img.close()



if __name__ == '__main__':
    source_fn_path = '../testResults/2022-10-19-22:51-come15k-train-data-analysis/wordTree.pkl'
    input_word = 'matter'
    save_dir = '..' + source_fn_path.strip(source_fn_path.split('/')[-1]) + 'SemanticDataset/' + input_word + "/"
    create_tree = create_tree(source_fn_path)
    output_dataset(input_word, create_tree, save_dir)