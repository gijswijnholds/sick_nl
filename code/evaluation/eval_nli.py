import os
import pickle
import torch
from sick_nl.code.config import (models_folder, results_folder, bert, bert_nl,
                                 roberta, roberta_nl, mbert)
from sick_nl.code.loaders.sick import load_sick_en, load_sick_nl
from sick_nl.code.loaders.nli_models import load_bert_nli_model


def save_model(out_folder, model, dataset_name, name, epoch=1):
    model_name = name.split('/')[-1]
    out_fn = os.path.join(out_folder, f"model_{dataset_name}_{model_name}_epoch{epoch}.pt")
    torch.save(model, out_fn)


def load_results(fn):
    with open(fn, 'rb') as in_file:
        results = pickle.load(in_file)
    return results


def save_results(out_folder, results, dataset_name, name, epoch=1):
    model_name = name.split('/')[-1]
    out_fn = os.path.join(out_folder, f"results_{dataset_name}_{model_name}_epoch{epoch}.p")
    with open(out_fn, 'wb') as out_file:
        pickle.dump(results, out_file)


def get_epoch(fn):
    return int(fn.split('epoch')[-1].split('.')[0])


def consolidate_results_and_models(dataset_name, name, model_folder, result_folder):
    model_name = name.split('/')[-1]
    results_fns = [os.path.join(result_folder, fn)
                   for fn in os.listdir(result_folder)
                   if f"{dataset_name}_{model_name}" in fn]
    print("Getting all results...")
    all_results = {get_epoch(fn): load_results(fn) for fn in results_fns}
    agg_result_folder = result_folder + '_agg'
    if not os.path.exists(agg_result_folder):
        os.mkdir(agg_result_folder)
    all_results_out_fn = os.path.join(agg_result_folder,
                                      f"results_{dataset_name}_{model_name}_epoch0-20.p")
    print("Saving all results...")
    with open(all_results_out_fn, 'wb') as out_file:
        pickle.dump(all_results, out_file)
    print("Removing superfluous result files...")
    for fn in results_fns:
        os.remove(fn)
    print("Getting best model...")
    best_epoch = sorted(list(all_results.items()),
                        key=lambda d: d[1]['eval_accuracy'], reverse=True)[0][0]
    best_model_fn = os.path.join(model_folder,
                                 f"model_{dataset_name}_{model_name}_epoch{best_epoch}.pt")
    best_model = torch.load(best_model_fn)
    agg_model_folder = model_folder + '_agg'
    if not os.path.exists(agg_model_folder):
        os.mkdir(agg_model_folder)
    best_model_out_fn = os.path.join(agg_model_folder,
                                     f"model_{dataset_name}_{model_name}_epoch{best_epoch}.pt")
    torch.save(best_model, best_model_out_fn)
    print("Removing superfluous models...")
    models_fns = [os.path.join(model_folder, fn)
                  for fn in os.listdir(model_folder)
                  if f"{dataset_name}_{model_name}" in fn]
    for fn in models_fns:
        os.remove(fn)


def run_finetuner(sick_dataset, name, setting='bert', num_epochs=3,
                  model_folder='models', result_folder='results'):
    tuner = load_bert_nli_model(sick_dataset, name, setting)
    eval_results = []
    train_results = []
    eval_results.append(tuner.evaluate())
    save_results(result_folder, eval_results[-1], sick_dataset.name, name, epoch=0)

    epochs = num_epochs
    for i in range(epochs):
        j = i+1
        print(f"Training for epoch {j}/{epochs}...")
        train_results.append(tuner.train())
        eval_results.append(tuner.evaluate())
        save_results(result_folder, eval_results[-1], sick_dataset.name, name, epoch=j)
        save_model(model_folder, tuner.model, sick_dataset.name, name, epoch=j)
    last_acc = eval_results[-1]['eval_accuracy']
    print(f"Finished running! The last test accuracy was {last_acc}!")
    consolidate_results_and_models(sick_dataset.name, name, model_folder, result_folder)
    quit()


def evaluate_en_nli_models():
    en_sick = load_sick_en()
    run_finetuner(en_sick, mbert, setting='bert', num_epochs=20, model_folder=models_folder, result_folder=results_folder)
    run_finetuner(en_sick, bert, setting='bert', num_epochs=20, model_folder=models_folder, result_folder=results_folder)
    run_finetuner(en_sick, roberta, setting='roberta', num_epochs=20, model_folder=models_folder, result_folder=results_folder)


def evaluate_nl_nli_models():
    nl_sick = load_sick_nl()
    run_finetuner(nl_sick, bert_nl, setting='bert', num_epochs=20, model_folder=models_folder, result_folder=results_folder)
    run_finetuner(nl_sick, roberta_nl, setting='roberta', num_epochs=20, model_folder=models_folder, result_folder=results_folder)
    run_finetuner(nl_sick, mbert, setting='bert', num_epochs=20, model_folder=models_folder, result_folder=results_folder)
