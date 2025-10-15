import json
import torch
import argparse

from easyeditor import BaseEditor, KoLEGHyperParams 

def main(alg_name,
        model_name,
        hparams_fname,
        ds_name,
        dataset_size_limit,
        log_dir,
        batch_size,
        retriever
        ):

    if dataset_size_limit is None: test_ds = json.load(open(ds_name, 'r', encoding='utf-8'))
    else: test_ds = json.load(open(ds_name, 'r', encoding='utf-8'))[:dataset_size_limit]

    hparams = KoLEGHyperParams.from_hparams(hparams_fname)

    modify_nums = {'①':'1', '②':'2', '③':'3', '④':'4', '⑤':'5', '⑥':'6', '⑦':'7', '⑧':'8', '⑨':'9', '⑩':'10', '⑪':'11', '⑫':'12', '⑬':'13', '⑭':'14', '⑮':'15', '⑯':'16', '⑰':'17', '⑱':'18', '⑲':'19', '⑳':'20', '⑴':'21', '⑵':'22', '⑶':'23', '⑷':'24', '⑸':'25', '⑹':'26', '⑺':'27', '⑻':'28', '⑼':'29', '⑽':'30', '⑾':'31', '⑿':'32', '〇':'0'}

    prompts = []
    ground_truth = []
    target_new = []
    subject = []
    rephrase_prompts = []
    locality_inputs = []
    locality_inputs = {
        'neighborhood':{
            'prompt': [],
            'ground_truth': []
        }}
    portability_inputs = {
        'neighborhood':{
            'prompt': [],
            'ground_truth': []
        }}
    time_str = []
    gold = []

    for i in test_ds:
            time = i['info']['시행기간'].split('-')
            info = str(time[0][:4]) + '년 ' + str(time[0][4:6]) + '월 ' + str(time[0][6:]) + '일부터 ' + str(time[1][:4]) + '년 ' + str(time[1][4:6]) + '월 ' + str(time[1][6:]) + '일까지 시행된 ' + i['info']["법령명"] + '의 ' +i['info']["조 번호"]+'조 '
            if i['info']["항 번호"] != '': info += i['info']["항 번호"]+'항 '
            if i['info']["호 번호"] != '': info += i['info']["호 번호"]+'호 '
            info = info[:-1] + '의 내용은 \"' + i['info']["content"] + '\"이다.'

            for key in modify_nums.keys():
                info = info.replace(key,modify_nums[key])
                
            gold.append(info)

            loc_time = i['info']["시행기간"].split('-')[0]
            loc_time_str = loc_time[:4] + '년 ' + loc_time[4:6] + '월 ' + loc_time[6:] + '일에 시행하고 있는 '

            time = i['info']["시행기간"].split('-')
            time_str.append(str(time[0][:4]) + '년 ' + str(time[0][4:6]) + '월 ' + str(time[0][6:]) + '일부터 ' + str(time[1][:4]) + '년 ' + str(time[1][4:6]) + '월 ' + str(time[1][6:]) + '일까지 시행된 ')

            if 'completion' in i: 
                prompts.append(i['completion'])
                subject.append(i['completion'])
                locality_inputs['neighborhood']['prompt'].append(loc_time_str+i['locality']['completion'])
                locality_inputs['neighborhood']['ground_truth'].append(i['locality']['answer'])
                portability_inputs['neighborhood']['prompt'].append(i['portability']['forward'])
                portability_inputs['neighborhood']['ground_truth'].append(i['portability']['backward'])
                rephrase_prompts.append(i['paraphrased_completion'])
            else:
                prompts.append(i['question'])
                subject.append(i['question'])
                locality_inputs['neighborhood']['prompt'].append(loc_time_str+i['locality']['question'])
                locality_inputs['neighborhood']['ground_truth'].append(i['locality']['answer'])
                portability_inputs['neighborhood']['prompt'].append(i['portability']['question'])
                portability_inputs['neighborhood']['ground_truth'].append(i['portability']['answer'])
                rephrase_prompts.append(i['paraphrased_question'])
            
            if 'a' not in i: ground_truth.append(i['answer'])
            else: ground_truth.append(i['a'])
            target_new.append(i['answer'])

    editor=BaseEditor.from_hparams(hparams)

    metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subject,
            train_ds=None,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            time_str=time_str,
            ds_name=ds_name,
            batch_size=batch_size,
            retriever=retriever,
            gold=gold
        )
    json.dump(metrics, open('results.json', 'w', encoding='utf-8'), indent=4, ensure_ascii = False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        default="KoLEG",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="llama.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/",
        )
    parser.add_argument(
        "--retriever",
        type=str,
        default="BAAI/bge-m3",
        )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.log_dir,
        args.batch_size,
        args.retriever
    )

