# import keras
import json
import os
from itertools import product

# from scripts.coverage import custom_objects


# folder_path = 'model_json'
# api_config_pool_path = 'api_config_pool.json'
PARAMETER_SPACE = 5


# def custom_objects():
#     def no_activation(x):
#         return x

#     def leakyrelu(x):
#         import keras.backend as K
#         return K.relu(x, alpha=0.01)

#     objects = {}
#     objects['no_activation'] = no_activation
#     objects['leakyrelu'] = leakyrelu
#     return objects


def extract_edges(model):
    layer_list = model.layers

    existing_edges = []
    for layer in layer_list:
        start_layer_class = layer.__class__.__name__
        if "Input" in start_layer_class:
            continue
        for node in layer._outbound_nodes:
            end_layer_class = node.outbound_layer.__class__.__name__
            edge = (start_layer_class, end_layer_class)  # edge should be direct
            if edge not in existing_edges:
                existing_edges.append(edge)
    return existing_edges


def extract_nodes(model):
    """
    existing_nodes: {"layer_name1": [layer_config1, layer_config2], "layer_name2": [], ...}
    """
    layer_list = model.layers
    existing_nodes = {}

    for layer in layer_list:
        layer_config = layer.get_config()
        layer_config.pop("name")
        if "filters" in layer_config: layer_config.pop("filters")
        if "units" in layer_config: layer_config.pop("units")
        layer_class = layer.__class__.__name__
        if 'Input' in layer_class:
            continue
        if layer_class not in existing_nodes:
            existing_nodes[layer_class] = []
        if layer_config not in existing_nodes[layer_class]:
            existing_nodes[layer_class].append(layer_config)
    return existing_nodes


def extract_inputs(model):
    """
    existing_inputs: {"layer_class": {"input_dims": [], "dtype": [], "shape": []}}
    layer_dims: {"layer_class": {"input_dims": [], "output_dims": []}}
    """
    # if model.__class__.__name__ == 'Sequential':
    #     layer_list = model.layers
    # else:
    #     layer_list = model.layers[1:]  # ignore the first input layer
    layer_list = model.layers
    existing_inputs = {}
    layer_dims = {}
    for layer in layer_list:
        layer_class = layer.__class__.__name__
        if 'Input' in layer_class:
            continue
        if layer_class not in existing_inputs:
            existing_inputs[layer_class] = {"input_dims": [], "dtype": [], "shape": []}
            layer_dims[layer_class] = {"input_dims": [], "output_dims": []}
        input_dims = len(layer.input.shape)
        output_dims = len(layer.output.shape)
        dtype = str(layer.input.dtype.name)
        shape = str(list(layer.input.shape))
        if input_dims not in existing_inputs[layer_class]['input_dims']:
            existing_inputs[layer_class]['input_dims'].append(input_dims)
        if input_dims not in layer_dims[layer_class]['input_dims']:
            layer_dims[layer_class]['input_dims'].append(input_dims)
        if output_dims not in layer_dims[layer_class]['output_dims']:
            layer_dims[layer_class]['output_dims'].append(output_dims)
        if dtype not in existing_inputs[layer_class]['dtype']:
            existing_inputs[layer_class]['dtype'].append(dtype)
        if shape not in existing_inputs[layer_class]['shape']:
            existing_inputs[layer_class]['shape'].append(shape)
    return existing_inputs, layer_dims


# def model_to_json(model_path,folder_path):
#     cur_model = keras.models.load_model(model_path, custom_objects=custom_objects())
#     existing_edges = extract_edges(cur_model)
#     existing_node = extract_nodes(cur_model)
#     existing_inputs, layer_dims = extract_inputs(cur_model)
#     layer_dict = {}
#     cur_edge_num = 0
#     for layer in cur_model.layers:
#         layer_name = layer.__class__.__name__
#         if 'Input' in layer_name:
#             continue
#         layer_dict[layer_name] = layer_dict[layer_name] + 1 if layer_name in layer_dict else 1
#         inbound_nodes = layer._inbound_nodes
#         if inbound_nodes:
#             if isinstance(inbound_nodes[0].inbound_layers, list):
#                 cur_edge_num += len(inbound_nodes[0].inbound_layers)
#             else:
#                 if inbound_nodes:
#                     cur_edge_num += 1

#     layer_num = sum(layer_dict.values())
#     layer_type = list(layer_dict.keys())
#     print(f'layer_num:{layer_num}')
#     print(f'layer_type_num:{layer_type}')
#     print(f'cur_edge_num:{cur_edge_num}')
#     print(existing_edges)
#     print(existing_node)
#     print(existing_inputs)
#     print(layer_dims)

#     cur_model_info = {}
#     cur_model_info['edges'] = existing_edges
#     cur_model_info['layer_config'] = existing_node
#     cur_model_info['layer_input_info'] = existing_inputs
#     cur_model_info['layer_num'] = layer_num
#     cur_model_info['layer_type'] = layer_type
#     cur_model_info['cur_edge_num'] = cur_edge_num
#     cur_model_info['layer_dims'] = layer_dims
#     json_path = os.path.join(folder_path, model_path.split("\\")[-1][:-3]+'.json')
#     # if not os.path.exists(json_path):
#     with open(json_path, 'w') as json_file:
#         json.dump(cur_model_info, json_file, indent=4)


def union_json(single_json_path, all_json_path):
    """
    single_json_path:单个json文件的路径
    all_json_path:分母json文件的路径
    """
    with open(single_json_path, 'r') as json_file:
        model_info = json.load(json_file)
    if not os.path.exists(all_json_path):
        all_layer_info = {}
    else:
        with open(all_json_path, 'r') as all_json_file:
            all_layer_info = json.load(all_json_file)

    if 'layer_config' not in all_layer_info.keys():
        all_layer_info['layer_config'] = {}
    for layer_class, layer_configs in model_info['layer_config'].items():
        if layer_class not in all_layer_info['layer_config'].keys():
            all_layer_info['layer_config'][layer_class] = layer_configs
        else:
            for layer_config in layer_configs:
                if layer_config not in all_layer_info['layer_config'][layer_class]:
                    all_layer_info['layer_config'][layer_class].append(layer_config)

    if 'layer_input_info' not in all_layer_info.keys():
        all_layer_info['layer_input_info'] = {}
    for layer_class, layer_input_info in model_info['layer_input_info'].items():
        if layer_class not in all_layer_info['layer_input_info'].keys():
            all_layer_info['layer_input_info'][layer_class] = layer_input_info
        else:
            for attr in ["input_dims", "dtype", "shape"]:
                if attr not in all_layer_info['layer_input_info'][layer_class].keys():
                    all_layer_info['layer_input_info'][layer_class][attr] = layer_input_info[attr]
                else:
                    all_layer_info['layer_input_info'][layer_class][attr] = list(
                        set(layer_input_info[attr]).union(set(all_layer_info['layer_input_info'][layer_class][attr])))

    if 'layer_dims' not in all_layer_info.keys():
        all_layer_info['layer_dims'] = {}
    for layer_class, layer_dims in model_info['layer_dims'].items():
        if layer_class not in all_layer_info['layer_dims'].keys():
            all_layer_info['layer_dims'][layer_class] = layer_dims
        else:
            for attr in ["input_dims", "output_dims"]:
                if attr not in all_layer_info['layer_dims'][layer_class].keys():
                    all_layer_info['layer_dims'][layer_class][attr] = layer_dims[attr]
                else:
                    all_layer_info['layer_dims'][layer_class][attr] = list(
                        set(layer_dims[attr]).union(set(all_layer_info['layer_dims'][layer_class][attr])))

    if 'layer_type' not in all_layer_info.keys():
        all_layer_info['layer_type'] = model_info['layer_type']
    else:
        all_layer_info['layer_type'] = list(set(model_info['layer_type']).union(set(all_layer_info['layer_type'])))

    if 'max_edge_num' not in all_layer_info.keys():
        all_layer_info['max_edge_num'] = model_info['cur_edge_num']
    else:
        all_layer_info['max_edge_num'] = max(all_layer_info['max_edge_num'], model_info['cur_edge_num'])

    if 'max_layer_num' not in all_layer_info.keys():
        all_layer_info['max_layer_num'] = model_info['layer_num']
    else:
        all_layer_info['max_layer_num'] = max(all_layer_info['max_layer_num'], model_info['layer_num'])

    with open(all_json_path, 'w') as json_file:
        json.dump(all_layer_info, json_file, indent=4)


class CoverageCalculatornew():

    # init里只能是和具体模型无关的数值的初始化
    def __init__(self, all_json_path,api_config_pool_path):
        self.all_layer_info = {}
        self.edges = []
        self.all_edges = []
        self.layer_config = {}
        self.layer_input_info = {}
        self.POSSIBLE_DTYPE = {'bfloat16', 'double', 'float16', 'float32', 'float64', 'half'}

        with open(api_config_pool_path, "r") as pool_file:
            self.api_config_pool = json.load(pool_file)
        with open(all_json_path, 'r') as json_file:
            self.all_layer_info = json.load(json_file)

        self.total_dtype_num = len(self.all_layer_info["layer_input_info"]) * len(self.POSSIBLE_DTYPE)
        self.total_shape_num = len(self.all_layer_info["layer_input_info"]) * PARAMETER_SPACE
        self.total_ndims_num = 0
        for layer_class in self.all_layer_info["layer_input_info"]:
            ndims_list = self.all_layer_info["layer_input_info"][layer_class]["input_dims"]
            self.total_ndims_num += len(ndims_list)
        self.total_input_num = self.total_ndims_num + self.total_dtype_num + self.total_shape_num
        print(self.total_ndims_num)
        self.total_param = {}
        # self.total_param_list = {}
        self.total_param_num = 0
        for layer_class in self.api_config_pool:
            self.total_param[layer_class] = 0
            # self.total_param_list[layer_class] = {}
            for config in self.api_config_pool[layer_class]:
                # self.total_param_list[layer_class][config] = []
                if self.api_config_pool[layer_class][config] == [0]:
                    self.total_param[layer_class] += PARAMETER_SPACE
                else:
                    self.total_param[layer_class] += len(self.api_config_pool[layer_class][config])
            self.total_param_num += self.total_param[layer_class]
        print(self.total_ndims_num)
        for pre_layer, next_layer in product(self.all_layer_info["layer_dims"].keys(), repeat=2):
            if set(self.all_layer_info["layer_dims"][pre_layer]["output_dims"]).intersection(
                    set(self.all_layer_info["layer_dims"][next_layer]["input_dims"])) != 0:
                self.all_edges.append([pre_layer, next_layer])

        self.max_edge_num = self.all_layer_info['max_edge_num']
        self.max_layer_num = self.all_layer_info['max_layer_num']
        self.layer_type = len(self.all_layer_info["layer_type"])

        self.cur_edge_num = 0
        self.cur_layer_num = 0
        self.cur_layer_type = 0
        print(len(self.all_edges))

    def load_json(self, json_path):
        with open(json_path, 'r') as json_file:
            model_info = json.load(json_file)
        print(model_info.keys())
        self.cur_edge_num = model_info["cur_edge_num"]
        self.cur_layer_num = model_info['layer_num']
        self.cur_layer_type = len(model_info['layer_type'])
        self.edges = []
        self.layer_config = {}
        self.layer_input_info = {}

        for edge in model_info['edges']:
            if edge not in self.edges:
                self.edges.append(edge)

        for class_type, configs in model_info["layer_config"].items():
            if class_type not in self.layer_config:
                self.layer_config[class_type] = configs
            else:
                for config in configs:
                    if config not in self.layer_config[class_type]:
                        self.layer_config[class_type].append(config)
        # print(self.layer_config)

        for layer_class, layer_input_info in model_info['layer_input_info'].items():
            if layer_class not in self.layer_input_info:
                self.layer_input_info[layer_class] = layer_input_info
            else:
                for attr in ["input_dims", "dtype", "shape"]:
                    if attr not in self.layer_input_info[layer_class].keys():
                        self.layer_input_info[layer_class][attr] = layer_input_info[attr]
                    else:
                        self.layer_input_info[layer_class][attr] = list(
                            set(layer_input_info[attr]).union(
                                set(self.layer_input_info[layer_class][attr])))

    def api_pair_coverage(self):
        print(f"The API Pair Coverage Is: {len(self.edges)}/{len(self.all_edges)}")
        return len(self.edges)/len(self.all_edges)

    def _layer_config_coverage(self, layer_config_list, layer_class):
        """
            hp: count of param_value.
            param_list: {param1: [value1, value2], ...}
        """
        config_pool = self.api_config_pool[layer_class]
        param_list = {}
        for param in config_pool:
            param_list[param] = []
        hp = 0
        # Journal Submitted Version is Below.
        for layer_config in layer_config_list:
            for param in layer_config:
                if param not in param_list:
                    continue
                if config_pool[param] == [0]:
                    if layer_config[param] not in param_list[param] and len(param_list[param]) <= PARAMETER_SPACE:
                        param_list[param].append(layer_config[param])
                        hp += 1
                else:
                    if layer_config[param] not in param_list[param]:
                        param_list[param].append(layer_config[param])
                        hp += 1
        return hp, param_list

    def config_coverage(self):
        total_hp = 0
        for layer_class in self.layer_config:
            if layer_class in self.api_config_pool:
                layer_config_list = self.layer_config[layer_class]
                hp, param_list = self._layer_config_coverage(layer_config_list, layer_class)
                total_hp += hp
        print(f"The Configuration Coverage is: {total_hp}/{self.total_param_num}")
        return total_hp / self.total_param_num

    def ndims_coverage(self):
        """
        ndims_cov
        """
        covered_ndims_num = 0
        for layer_class in self.layer_input_info:
            ndims_list = self.layer_input_info[layer_class]["input_dims"]
            covered_ndims_num += len(ndims_list)
        return covered_ndims_num

    def dtype_coverage(self):
        covered_dtype_num = 0
        for layer_class in self.layer_input_info:
            dtype_list = self.layer_input_info[layer_class]["dtype"]
            covered_dtype_num += len(dtype_list)
        return covered_dtype_num

    def shape_coverage(self):
        covered_shape_num = 0
        for layer_class in self.layer_input_info:
            shape_list = self.layer_input_info[layer_class]["shape"]
            covered_shape_num += min(len(shape_list),
                                     PARAMETER_SPACE)  # if the total number of shape is larger that SHAPE_SPACE, we set it as 100%
        return covered_shape_num

    def input_coverage(self):
        """
        input_cov = ndim_cov + dtype_cov + shape_cov
        """
        covered_ndims = self.ndims_coverage()
        covered_dtype = self.dtype_coverage()
        covered_shape = self.shape_coverage()
        print(f"The NDims Coverage Is: {covered_ndims}/{self.total_ndims_num}")
        print(f"The DType Coverage Is: {covered_dtype}/{self.total_dtype_num}")
        print(f"The Shape Coverage Is: {covered_shape}/{self.total_shape_num}")
        print(f"The Input Coverage Is: {covered_ndims + covered_dtype + covered_shape}/{self.total_input_num}")
        input_cov = (covered_ndims + covered_dtype + covered_shape) / self.total_input_num
        ndims_cov = covered_ndims / self.total_ndims_num
        dtype_cov = covered_dtype / self.total_dtype_num
        shape_cov = covered_shape / self.total_shape_num
        return input_cov, ndims_cov, dtype_cov, shape_cov

    def op_type_cover(self):
        print(f'op_type_cover is: {self.cur_layer_type}/{self.layer_type}')
        return self.cur_layer_type/self.layer_type

    def op_num_cover(self):
        print(f'op_num_cover is: {self.cur_layer_num}/{self.max_layer_num}')
        return self.cur_layer_num / self.max_layer_num

    def edge_cover(self):
        print(f'edge_cover is: {self.cur_edge_num}/{self.max_edge_num}')
        return self.cur_edge_num / self.max_edge_num

    def cal_coverage(self):
        input_cov, ndims_cov, dtype_cov, shape_cov=self.input_coverage()
        config_cov=self.config_coverage()
        api_cov=self.api_pair_coverage()
        op_type_cov=self.op_type_cover()
        op_num_cov=self.op_num_cover()
        edge_cov=self.edge_cover()
        
        return input_cov,config_cov,api_cov,op_type_cov,op_num_cov,edge_cov

if __name__ == '__main__':
    # model_path1 = 'data/mnist_output/000005/models/tensorflow.h5'
    # model_path2 = 'data/mnist_output/000004/models/tensorflow.h5'
    #  F:\NEW\比赛\项目\MR2023 - master\torch_mutated_net\vgg11\2023_12_16_20_45_30\model_json
    all_json_path="/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/ms_all_layer_info.json"
    api_config_pool_path = '/home/cvgroup/myz/czx/SemTest_master/mindspore_mutation/mindspore_api_config_pool.json'
    # model_to_json(model_path1)
    # model_to_json(model_path2)

    # for file in os.listdir(folder_path):
    #     if file != 'all_layer_info.json':
    #         file_path = os.path.join(folder_path, file)
    #         union_json(file_path, all_json_path)

    cal_cov = CoverageCalculatornew(all_json_path,api_config_pool_path)
    # cal_cov.load_json('./torch_mutated_net/vgg11/2023_12_16_20_45_30/model_json/model_99.json')
    # cal_cov.cal_coverage()
