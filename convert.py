# convert the model
import argparse


def bigdl_llm_convert(family,model_path,output_model_path,outtype):
    from bigdl.llm import llm_convert
    bigdl_llm_path = llm_convert(model=model_path,
                                 outfile=output_model_path, 
                                 outtype=outtype, 
                                 model_family=family)
    return bigdl_llm_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script to convert model')
    parser.add_argument('-t', '--type', type=str, choices=["bigdl-llm"],
                        help='select a convert type')
    parser.add_argument('-d', '--dtype', type=str, choices=["int4", "int8"],
                        help='output datatype')
    parser.add_argument('-f', '--family', type=str, choices=["llama", "chatglm"],
                        help='select a model family')
    parser.add_argument('-i', '--input-model-path', type=str,
                        help='input model path')
    parser.add_argument('-o', '--output-model-dir', type=str,
                        help='output model diractory')


    args = parser.parse_args()
    type = args.type
    out_dtype = args.dtype
    family = args.family
    model_path = args.input_path
    output_model_path = args.output_model_path
    if type == "bigdl-llm":
        print(bigdl_llm_convert(family,model_path, output_model_path,out_dtype))