USAGE_DOC="""
Calculate change of stability (ddG) using FoldX
for a mutated sequence.

Required input file:
A TSV describing sequence and other meta information:
- Column `PDB`: PDB ID (or path to a PDB file)
- Column `Chain`: Chain ID (e.g., A, B, C)
- Column `WT_seq`: wildtype sequence
- Column `MT_seq`: mutated sequence
- Column `Start_index` [Optional]: Index (1-based) of the first AA in `WT_seq`
    with respect to the full WT sequence in PDB. This optional column is used to
    keep the TSV file small. For example, if you have a sequence with 1000AA
    and only want to mutate the range [100, 150], you can only copy-paste
    100-150 AAs into `WT_seq` and set `Start_index` to 100. If not provided,
    it is assumed to be 1.

Example:
python foldx_stability_eval.py \\
-i /export/share/yunan/tmp/foldx_eval/run_foldx_ace2.tsv \\
-o /export/share/yunan/tmp/foldx_eval/run_foldx_ace2-output
"""
import pandas as pd
import subprocess
import multiprocessing
import argparse
import pathlib
import tqdm
import time
import os


class FoldXWrapper(object):
    def __init__(self, input_tsv=None, save_dir=None, 
            n_job=1, foldx_runs=5, foldx='foldx', keep_files=False, repair_pdb_dir=None):
        self.meta = pd.read_table(input_tsv)
        print("self.meta.iloc[:3]: ", self.meta.iloc[:3])
        self.save_dir = pathlib.Path(save_dir)
        if repair_pdb_dir is not None:
            self.repair_pdb_dir = pathlib.Path(repair_pdb_dir)
            self.repair_pdb_dir.mkdir(parents=True, exist_ok=True)
            print("make repair_pdb_dir")
        else:
            self.repair_pdb_dir = None
        self.n_job = n_job
        self.foldx_runs = foldx_runs
        self.foldx = foldx
        self.keep_files = keep_files
        self._pdb_cache_dir = self.save_dir/'pdb_cache'
        self.setup_directory()
    
    def setup_directory(self):
        """
        Create saving directories
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._pdb_cache_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, len(self.meta) + 1):
            job_dir = self.save_dir/f'job_{i}'
            job_dir.mkdir(parents=True, exist_ok=True)
    
    def clean(self):
        """
        Clean intermediate files
        """
        subprocess.run(['rm', '-r', self._pdb_cache_dir])
        for i in range(1, len(self.meta) + 1):
            job_dir = self.save_dir/f'job_{i}'
            subprocess.run(['rm', '-r', job_dir])
    
    def clean_batch(self, start, end):
        """
        Clean intermediate files
        """
        # subprocess.run(['rm', '-r', self._pdb_cache_dir])
        for i in range(start+1, end + 1):
        # for i in range(1, len(self.meta) + 1):
            # print("clean_batch i:", i) 
            job_dir = self.save_dir/f'job_{i}'
            # print("clean_batch job_dir:", job_dir) 
            subprocess.run(['rm', '-r', job_dir])

    def prepare_job_list(self):
        """
        Read input TSV and generate FoldX-format input file.
        """
        job_list = []
        for job_id, row in self.meta.iterrows():
            pdb_id_or_file = row['PDB']
            chain = row['Chain']
            wt_seq = row['WT_seq']
            mt_seq = row['MT_seq']
            offset = 0 if 'Start_index' not in row else row['Start_index'] - 1
            job_dir = self.save_dir/f'job_{job_id + 1}'

            # setup PDB
            if pathlib.Path(pdb_id_or_file).is_file():
                # a PDB path is provided
                pdb_file = pathlib.Path(pdb_id_or_file)
                pdb_dir = pdb_file.parent
                pdb_id = pdb_file.stem
            else:
                # a PDB ID is provided
                pdb_dir = self._pdb_cache_dir
                pdb_id = pdb_id_or_file
                pdb_file = self._pdb_cache_dir/f'{pdb_id}.pdb'
                if not pdb_file.exists():
                    #  try dowloading if not downloaded yet
                    try:
                        subprocess.run([
                            'wget',
                            '-q',
                            f'https://files.rcsb.org/download/{pdb_id}.pdb',
                            '-P', self._pdb_cache_dir
                        ])
                    except Exception as e:
                        print(f'Error when downloading PDB: {e}')
            
            # setup mutation file
            mutations = []
            mutation_list_file = job_dir/'individual_list.txt'
            for i, (ref, alt) in enumerate(zip(wt_seq, mt_seq), start=1):
                if ref != alt:
                    mutations.append(f'{ref}{chain}{i + offset}{alt}')
            with open(mutation_list_file, 'w') as fout:
                fout.write(','.join(mutations) + ';')
            
            # setup running params
            params = {
                'pdb_dir': pdb_dir,
                'pdb_id': pdb_id,
                'job_dir': job_dir,
                'mutation_list': mutation_list_file,
                'repair_pdb_dir': self.repair_pdb_dir,
            }
            job_list.append(params)
        return job_list
    
    def run_foldx(self, params):
        pdb_dir = params['pdb_dir']
        pdb_id = params['pdb_id']
        job_dir = params['job_dir']
        repair_pdb_dir = params['repair_pdb_dir']
        mutation_list = params['mutation_list']

        # print("pdb_dir: ", pdb_dir)
        # print("pdb_id: ", pdb_id)
        # print("job_dir: ", job_dir)
        # print("repair_pdb_dir: ", repair_pdb_dir)
        # print("mutation_list: ", mutation_list)

        # print("cmd: ", [self.foldx, '--command=RepairPDB', 
        #                 f'--pdb-dir={pdb_dir}',
        #                 f'--pdb={pdb_id}.pdb',
        #                 f'--output-dir={job_dir}',
        #                 '--screen=false'
        #                 ])

        if repair_pdb_dir is None:
            # Repair
            subprocess.call([self.foldx, '--command=RepairPDB', 
                            f'--pdb-dir={pdb_dir}',
                            f'--pdb={pdb_id}.pdb',
                            f'--output-dir={job_dir}',
                            '--screen=false'
                            ])

            # BuildModel
            subprocess.call([self.foldx, '--command=BuildModel',
                            f'--pdb-dir={job_dir}',
                            f'--pdb={pdb_id}_Repair.pdb',
                            f'--mutant-file={mutation_list}',
                            f'--output-dir={job_dir}',
                            f'--numberOfRuns={self.foldx_runs}',
                            '--screen=false'])
        
        else:
            repaired_pdb_path = os.path.join(repair_pdb_dir, f'{pdb_id}_Repair.pdb')
            repaired_fxout_path = os.path.join(repair_pdb_dir, f'{pdb_id}_Repair.fxout')
            print("repair_pdb_dir: ", repair_pdb_dir)
            print("repaired_pdb_path: ", repaired_pdb_path)
            print("A os.path.isfile(repaired_pdb_path): ", os.path.isfile(repaired_pdb_path))
            if not os.path.isfile(repaired_pdb_path) or not os.path.isfile(repaired_fxout_path):
                print("Running RepairPDB...")
                # Repair
                subprocess.call([self.foldx, '--command=RepairPDB', 
                                f'--pdb-dir={pdb_dir}',
                                f'--pdb={pdb_id}.pdb',
                                f'--output-dir={repair_pdb_dir}',
                                '--screen=false'
                                ])
                print("RepairPDB done!")

            # print("B os.path.isfile(repaired_pdb_path): ", os.path.isfile(repaired_pdb_path))
            # BuildModel
            subprocess.call([self.foldx, '--command=BuildModel',
                            f'--pdb-dir={repair_pdb_dir}',
                            f'--pdb={pdb_id}_Repair.pdb',
                            f'--mutant-file={mutation_list}',
                            f'--output-dir={job_dir}',
                            f'--numberOfRuns={self.foldx_runs}',
                            '--screen=false'])
    
    def merge_results(self):
        """
        Pull ddG for every job and create a summary report
        """
        ddG_list = []
        for i in range(1, len(self.meta) + 1):
            job_dir = self.save_dir/f'job_{i}'
            print("job_dir: ", job_dir)
            files = list(job_dir.glob('Average_*.fxout'))
            print("files: ", files)
            avg_file = files[0]
            avg_df = pd.read_table(avg_file, skiprows=8)
            ddG = avg_df['total energy'].values[0]
            ddG_list.append(ddG)
        res = self.meta.copy() # create meta batches here to save
        try:
            res['ddG'] = ddG_list
        except:
            print(res.shape, len(ddG_list))
        return res
        
    def merge_results_batch(self, start, end):
        """
        Pull ddG for every job and create a summary report
        """
        ddG_list = []
        print("merge_results_batch start +1: ", start+1)
        print("merge_results_batch end + 1: ", end + 1)
        res = self.meta.copy().iloc[start:end] # create meta batches here to save
        for i in range(start+1, end + 1): # 
        # for i in range(1, len(self.meta) + 1):
            # print("merge_results_batch i: ", i)
            job_dir = self.save_dir/f'job_{i}'
            # print("merge_results_batch job_dir: ", job_dir)
            files = list(job_dir.glob('Average_*.fxout'))
            # print("merge_results_batch files: ", files)
            if len(files) == 0:
                res = res.drop([i], errors='ignore')
                continue
            avg_file = files[0]
            avg_df = pd.read_table(avg_file, skiprows=8)
            ddG = avg_df['total energy'].values[0]
            ddG_list.append(ddG)
        print("A res: ", res)
        print("B len(ddG_list): ", len(ddG_list))
        print("B ddG_list): ", ddG_list)
        # breakpoint()
        res['ddG'] = ddG_list
        #except:
        #    print(res.shape, len(ddG_list))
        # print("B res: ", res)
        return res
    
    # def run(self):
    #     start_time = time.time()
    #     job_list = self.prepare_job_list()
    #     pool = multiprocessing.Pool(args.workers)
    #     print("len(job_list): ", len(job_list))
    #     print("job_list[:10]: ", job_list[:10])
    #     print("job_list: ", job_list)
    #     for _ in tqdm.tqdm(pool.imap_unordered(self.run_foldx, job_list),
    #             total=len(job_list)):
    #         pass
    #     pool.close()
    #     pool.join()

    #     end_time = time.time()
    #     print("Time taken for FoldX: ", end_time - start_time)

    #     results = self.merge_results()
    #     results.to_csv(self.save_dir/'results.tsv', 
    #         index=False, sep='\t', float_format='%.6f')
    #     if not self.keep_files:
    #         print("cleaning files")
    #         self.clean()
    
    def run_batches(self, foldx_batch_size, start_batch_ind=0):
        print("start run_batches!")
        job_list = self.prepare_job_list()
        # print("len(job_list): ", len(job_list))
        # print("job_list[:10]: ", job_list[:10])
        # print("job_list: ", job_list)

        num_batches = -(-len(job_list)//foldx_batch_size)
        # print("num_batches: ", num_batches)
        # splitted = []
        len_l = len(job_list)
        for i in range(num_batches):
            
            print("run_batches # ", i)
            start = int(i*len_l/num_batches)
            end = int((i+1)*len_l/num_batches)
            print("start ind: ", start)
            print("end ind: ", end)
            # print("start: ", start, i*len_l/num_batches)
            # print("end: ", end, (i+1)*len_l/num_batches)
            # splitted.append(job_list[start:end])
            batch_job_list = job_list[start:end]

            # print("len(self.meta): ", len(self.meta))
            # print("self.meta: ", self.meta)
            # print("len(batch_job_list): ", len(batch_job_list))
            # print("batch_job_list: ", batch_job_list)
            # print("len(self.meta.iloc[start:end]): ", len(self.meta.iloc[start:end]))
            # print("self.meta.iloc[start:end]: ", self.meta.iloc[start:end])
        # for batch_job_list in tqdm.tqdm(splitted):
        #     print("batch_job_list: ", batch_job_list) 

            if i < start_batch_ind:
                print("skipping batch # ", i)
                continue
            else:
                start_time = time.time()
                pool = multiprocessing.Pool(args.workers)
                for _ in tqdm.tqdm(pool.imap_unordered(self.run_foldx, batch_job_list),
                        total=len(batch_job_list)):
                    pass
                pool.close()
                pool.join()

                end_time = time.time()
                print("Time taken for FoldX batch: ", end_time - start_time)

                # results = self.merge_results()
                results = self.merge_results_batch(start, end)
                # results.to_csv(self.save_dir/'results.tsv', 
                #     index=False, sep='\t', float_format='%.6f')
                results.to_csv(self.save_dir/f'results_batch{i}_{start}to{end-1}.tsv', 
                    index=False, sep='\t', float_format='%.6f')
            print("do clean_batch", start, end)
            if not self.keep_files:
                print("cleaning files")
                # self.clean()
                self.clean_batch(start, end)
        
        # merge results from batch tsv files into ONE full tsv file
        results_tsv_files = list(self.save_dir.glob('results_batch*to*.tsv'))
        # print("results_tsv_files: ", results_tsv_files)
        results_tsv_df_full = None
        print("Merging result tsv files")
        for tsv_ind, results_tsv_file in enumerate(results_tsv_files):
            print("tsv_ind: ", tsv_ind) 
            print("results_tsv_file: ", results_tsv_file)
            if tsv_ind == 0:
                results_tsv_df_full = pd.read_table(results_tsv_file)
            else:
                results_tsv_df = pd.read_table(results_tsv_file)
                results_tsv_df_full = results_tsv_df_full.append(results_tsv_df, ignore_index=True)

        results_tsv_df_full.to_csv(self.save_dir/f'results_full.tsv', 
            index=False, sep='\t', float_format='%.6f')
        print("Output full tsv file: ", self.save_dir/f'results_full.tsv')
            

def main():
    runner = FoldXWrapper(
        input_tsv=args.i, save_dir=args.o,
        n_job=args.workers, foldx_runs=args.foldx_runs, 
        foldx=args.foldx, repair_pdb_dir=args.repair_pdb_dir,
        keep_files = args.keep
    )
    # runner.run()
    runner.run_batches(args.foldx_batch_size, args.start_batch_ind)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, usage=USAGE_DOC)
    parser.add_argument('-i', action='store', help='input TSV file')
    parser.add_argument('-o', action='store', help='output directory')
    parser.add_argument('--foldx_batch_size', action='store', type=int, default=10, help='number of foldx inferences per batch of saving into tsv')
    parser.add_argument('--start_batch_ind', action='store', type=int, default=0, help='batch index to start FoldX inference, used to continue from previously interrputed inference')
    parser.add_argument('--repair_pdb_dir', action='store', type=str, default='./repaired', help='output directory for repair pdb')
    # parser.add_argument('--repair_pdb_dir', action='store', type=str, default=None, help='output directory for repair pdb')
    parser.add_argument('--workers', action='store', type=int, default=1, help='number of parallel jobs')
    parser.add_argument('--foldx_runs', action='store', type=int, default=5, help='number of runs in FoldX')
    parser.add_argument('--foldx', action='store', default='./foldx/foldx',
                         help='path to FoldX executable')
    parser.add_argument('--keep', action='store_true', help='do not delete intermediate FoldX files')
    args = parser.parse_args()
    main()
