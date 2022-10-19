import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from treelib import Tree
from collections import namedtuple


class TreeNodeData:
    def __init__(self, count, files_path_list):
        self.count = count,
        self.files_path_list = files_path_list


def get_object_hypernym_paths(input_word):
    input_object = wn.synsets(input_word, pos=wn.NOUN)
    hypernyms_paths = input_object[0].hypernym_paths()
    return hypernyms_paths


def build_tree(input_words, tree, file_path, error_list):
    input_words = input_words.split(',')[0].replace(' ', '_')
    paths_result_list = get_object_hypernym_paths(input_words)
    try:
        if len(paths_result_list) > 0:
            # 选择最短路径
            if len(paths_result_list) > 1:
                length = len(paths_result_list[0])
                final_result = paths_result_list[0]
                for result in paths_result_list:
                    if len(result) < length:
                        final_result = result
            else:
                final_result = paths_result_list[0]

        # 父节点名称
        father_node_id_for_next_node = ''
        for idx, element in enumerate(final_result):
            name = element.name().split('.')[0]
            if not tree.contains(name):
                # 为根节点
                tree_node_data = TreeNodeData(count=1, files_path_list=[file_path])
                if len(father_node_id_for_next_node) == 0 and idx == 0:
                    tree.create_node(tag=name, identifier=name, data=tree_node_data)
                    father_node_id_for_next_node = name
                else:
                    tree.create_node(tag=name, identifier=name, parent=father_node_id_for_next_node,
                                     data=tree_node_data)
                    father_node_id_for_next_node = name
            else:
                node_data = tree.get_node(nid=name).data
                new_count = node_data.count[0] + 1
                node_data.files_path_list.append(file_path)
                tree_node_data = TreeNodeData(count=new_count, files_path_list=node_data.files_path_list)
                tree.update_node(nid=name, data=tree_node_data)
                father_node_id_for_next_node = name
        # tree.show()
    except:
        error_list.append(file_path + "/" + input_words)



if __name__ == '__main__':
    result_list2 = get_object_hypernym_paths()
    path_result_list = get_object_hypernym_paths("neck_brace")
    tree = Tree()
    build_tree("penis", tree, '../123455')
    # build_tree(tree, result_list2, '../1235')
    tree.show(data_property="files_path_list")
