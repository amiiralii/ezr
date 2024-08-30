import os
import baseline
import ezr

for dir in os.listdir('data/'):
    if dir.find('.') == -1:
        for dataset in os.listdir(f'data/{dir}'):
            if dataset.split('.')[-1] == 'csv':
                
                if not os.path.exists(f'results/{dir}/{dataset[:-4]}/linear-{dataset}'):
                    baseline.calc_baseline('linear', f'data/{dir}/{dataset}')
                if not os.path.exists(f'results/{dir}/{dataset[:-4]}/lightgbm-{dataset}'):
                    baseline.calc_baseline('lightgbm', f'data/{dir}/{dataset}')
                    #print(f"Baseline for {dataset} calculated.")

                if not os.path.exists(f'results/{dir}/{dataset[:-4]}/dist-{dataset}'):
                    res = ezr.eg.regression(f'data/{dir}/{dataset}', 0)
                    baseline.export(res[0], res[1], res[2], f'data/{dir}/{dataset}', 'dist')
                    print('0', end = ', ')

                if not os.path.exists(f'results/{dir}/{dataset[:-4]}/prob-{dataset}'):
                    res = ezr.eg.regression(f'data/{dir}/{dataset}', 1)
                    baseline.export(res[0], res[1], res[2], f'data/{dir}/{dataset}', 'prob')
                    print('1', end = ', ')

                if not os.path.exists(f'results/{dir}/{dataset[:-4]}/prob-syn-{dataset}'):
                    res = ezr.eg.regression(f'data/{dir}/{dataset}', 2)
                    baseline.export(res[0], res[1], res[2], f'data/{dir}/{dataset}', 'prob-syn')
                    print('2')

                print(f"All experiments for {dataset} calculated.")