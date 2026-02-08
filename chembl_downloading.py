import aiohttp
import asyncio
import pandas as pd
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from typing import List, Tuple, Optional
import logging
from tdc.single_pred import ADME,Tox

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
The input file containing ChEMBL IDs should be a plain text (.txt) file.
The file should be formatted as follows:
- Each line should contain exactly one ChEMBL ID.
- No extra spaces or empty lines should be included.
- The file should have a .txt extension.

Example of a valid input file:
-------------------------------
CHEMBL25
CHEMBL192
CHEMBL1234567
CHEMBL345
-------------------------------

Note: Other file formats such as .csv, .xlsx, or .json are not supported.
Ensure that the file is saved as .txt for proper processing.

Example commands to run the script:
-----------------------------------

1. Using a .txt file with ChEMBL IDs:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --assay_type=B --pchembl_threshold_for_download=6.0 --output_file=activity_data.csv

2. Specifying multiple ChEMBL IDs directly in the command:
   python chembl_downloading.py --target_id=CHEMBL25,CHEMBL192 --assay_type=B --pchembl_threshold_for_download=6.0 --output_file=activity_data.csv

3. Combining a .txt file with additional specified ChEMBL IDs:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --target_id=CHEMBL345 --assay_type=B --pchembl_threshold_for_download=6.0 --output_file=activity_data.csv

4. Specifying a custom output file name:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --output_file=custom_output.csv

5. Limiting the number of CPU cores used:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --max_cores=4 --output_file=activity_data.csv

6. Changing the assay type filter:
   python chembl_downloading.py --smiles_input_file=/path/to/chembl_ids.txt --assay_type=A --output_file=activity_data.csv
"""

class ChEMBLDownloader:
    def __init__(self, max_concurrent=50, timeout=30):
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def create_session(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        return aiohttp.ClientSession(connector=connector, timeout=self.timeout)

    async def fetch_activities_async(self, session: aiohttp.ClientSession, target_ids: List[str], 
                                   assay_types: List[str], pchembl_threshold_for_download: float) -> pd.DataFrame:
        """Async version of fetch_activities with concurrent pagination"""
        logger.info(f"Starting to fetch activities for {len(target_ids)} targets from ChEMBL...")
        
        base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            'target_id__in': ','.join(target_ids),
            'assay_type__in': ','.join(assay_types),
            'pchembl_value__isnull': 'false',
            'only': 'molecule_chembl_id,pchembl_value,target_id,bao_label'
        }
        
        # First request to get total count and setup pagination
        async with self.semaphore:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch data. HTTP Status Code: {response.status}")
                    return pd.DataFrame()
                
                data = await response.json()
                
        if 'activities' not in data:
            logger.info("No activities found.")
            return pd.DataFrame()
            
        activities = data['activities']
        
        # If there are more pages, fetch them concurrently
        if 'page_meta' in data and data['page_meta']['total_count'] > data['page_meta']['limit']:
            total_count = data['page_meta']['total_count']
            limit = data['page_meta']['limit']
            
            # Create tasks for remaining pages
            tasks = []
            offset = limit
            while offset < total_count:
                page_params = params.copy()
                page_params['offset'] = offset
                tasks.append(self._fetch_page(session, base_url, page_params))
                offset += limit
            
            # Execute all page requests concurrently
            if tasks:
                page_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in page_results:
                    if isinstance(result, list):
                        activities.extend(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Page fetch failed: {result}")

        if activities:
            df = pd.DataFrame(activities)
            if 'pchembl_value' in df.columns:
                df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
                df = df[df['pchembl_value'].notnull() & (df['pchembl_value'] >= pchembl_threshold_for_download)]
                df.drop(columns=['bao_label'], errors='ignore', inplace=True)
            else:
                logger.warning("pchembl_value column not found.")
                return pd.DataFrame()
        else:
            df = pd.DataFrame()
        
        logger.info(f"Finished fetching activities. Retrieved {len(df)} records.")
        return df

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str, params: dict) -> List[dict]:
        """Fetch a single page of activities"""
        async with self.semaphore:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('activities', [])
                    else:
                        logger.warning(f"Failed to fetch page with offset {params.get('offset', 0)}")
                        return []
            except Exception as e:
                logger.warning(f"Exception fetching page: {e}")
                return []

    async def fetch_smiles_async(self, session: aiohttp.ClientSession, compound_id: str) -> Tuple[str, Optional[str]]:
        """Async version of fetch_smiles"""
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{compound_id}.json"
        
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch data for {compound_id}. Status: {response.status}")
                        return compound_id, None
                        
                    data = await response.json()
                    if not data:
                        logger.warning(f"No data returned for {compound_id}")
                        return compound_id, None
                        
                    molecule_structures = data.get('molecule_structures')
                    if not molecule_structures:
                        logger.warning(f"No molecule structures found for {compound_id}")
                        return compound_id, None
                        
                    smiles = molecule_structures.get('canonical_smiles')
                    return compound_id, smiles
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {compound_id}. Error: {e}")
                return compound_id, None

    async def check_and_download_smiles_async(self, compound_ids: List[str]) -> List[Tuple[str, str]]:
        """Async version of check_and_download_smiles with concurrent requests"""
        logger.info(f"Starting to download SMILES for {len(compound_ids)} compounds...")
        
        async with await self.create_session() as session:
            tasks = [self.fetch_smiles_async(session, compound_id) for compound_id in compound_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        smiles_data = []
        for result in results:
            if isinstance(result, tuple) and result[1]:  # compound_id, smiles
                smiles_data.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"SMILES fetch failed: {result}")
        
        logger.info(f"Finished downloading SMILES. Retrieved {len(smiles_data)} valid SMILES.")
        return smiles_data

    async def fetch_all_protein_targets_async(self) -> List[str]:
        """Async version of fetch_all_protein_targets with concurrent pagination"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_file = os.path.join(base_dir, 'training_files', 'all_protein_targets.txt')
        
        # Check if cached file exists
        if os.path.exists(cache_file):
            logger.info(f"Loading protein targets from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                targets = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(targets)} protein targets from cache")
            return targets
        
        logger.info("Fetching all protein targets from ChEMBL...")
        base_url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
        params = {
            'target_type': 'SINGLE PROTEIN',
            'only': 'target_id'
        }
        
        async with await self.create_session() as session:
            # First request to get pagination info
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch targets. Status: {response.status}")
                    return []
                    
                data = await response.json()
                
            if 'targets' not in data:
                logger.info("No targets found.")
                return []
                
            targets = [t['target_id'] for t in data['targets']]
            
            # If there are more pages, fetch them concurrently
            if 'page_meta' in data and data['page_meta']['total_count'] > data['page_meta']['limit']:
                total_count = data['page_meta']['total_count']
                limit = data['page_meta']['limit']
                
                # Create tasks for remaining pages
                tasks = []
                offset = limit
                while offset < total_count:
                    page_params = params.copy()
                    page_params['offset'] = offset
                    tasks.append(self._fetch_targets_page(session, base_url, page_params))
                    offset += limit
                
                # Execute all page requests concurrently
                if tasks:
                    page_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in page_results:
                        if isinstance(result, list):
                            targets.extend(result)
                        elif isinstance(result, Exception):
                            logger.warning(f"Target page fetch failed: {result}")
        
        logger.info(f"Found {len(targets)} protein targets")
        
        # Save targets to cache file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            f.write('\n'.join(targets))
        logger.info(f"Saved protein targets to: {cache_file}")
        
        return targets

    async def _fetch_targets_page(self, session: aiohttp.ClientSession, url: str, params: dict) -> List[str]:
        """Fetch a single page of targets"""
        async with self.semaphore:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [t['target_id'] for t in data.get('targets', [])]
                    else:
                        logger.warning(f"Failed to fetch targets page with offset {params.get('offset', 0)}")
                        return []
            except Exception as e:
                logger.warning(f"Exception fetching targets page: {e}")
                return []

def read_chembl_ids_from_file(file_path):
    if os.path.exists(file_path):
        logger.info(f"Reading ChEMBL IDs from {file_path}...")
        with open(file_path, 'r') as file:
            chembl_ids = [line.strip() for line in file.readlines() if line.strip()]
            return chembl_ids
    else:
        logger.error(f"File {file_path} does not exist.")
        return []

async def download_target_async(args):
    """Async version of download_target with concurrent processing"""

    if args.dataset == "tdc_adme":
        data = ADME(name = args.target_id,path = os.path.join("training_files","target_training_datasets",args.target_id))
        return        
    elif args.dataset == "tdc_tox":
        data = Tox(name = args.target_id,path = os.path.join("training_files","target_training_datasets",args.target_id))
        return
    elif args.dataset == "tdc_benchmark":
        return
    downloader = ChEMBLDownloader(max_concurrent=args.max_concurrent)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_ids = []
    
    if args.all_proteins:
        target_ids = await downloader.fetch_all_protein_targets_async()
    else:
        if args.target_id:
            target_ids.extend(args.target_id.split(','))
        if args.smiles_input_file:
            file_chembl_ids = read_chembl_ids_from_file(args.smiles_input_file)
            target_ids.extend(file_chembl_ids)
    
    assay_types = args.assay_type.split(',')
    
    # Process targets in batches for better memory management
    batch_size = args.target_process_batch_size
    
    async with await downloader.create_session() as session:
        for i in range(0, len(target_ids), batch_size):
            batch = target_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(target_ids) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            tasks = []
            for chembl_id in batch:
                output_dir = os.path.join(base_dir, 'training_files', 'target_training_datasets', chembl_id)
                output_path = os.path.join(output_dir, args.output_file)
                
                if os.path.exists(output_path):
                    logger.info(f"File {output_path} already exists. Skipping download.")
                    continue
                    
                tasks.append(process_single_target(downloader, session, chembl_id, assay_types, 
                                                 args.pchembl_threshold_for_download, output_dir, 
                                                 args.output_file))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

async def process_single_target(downloader: ChEMBLDownloader, session: aiohttp.ClientSession, 
                              chembl_id: str, assay_types: List[str], pchembl_threshold: float,
                              output_dir: str, output_file: str):
    """Process a single target asynchronously"""
    try:
        data = await downloader.fetch_activities_async(session, [chembl_id], assay_types, pchembl_threshold)
        
        if not data.empty:
            compound_ids = data['molecule_chembl_id'].unique().tolist()
            smiles_data = await downloader.check_and_download_smiles_async(compound_ids)
            
            if smiles_data:
                # Only create directory if there is data to save
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                smiles_df = pd.DataFrame(smiles_data, columns=["molecule_chembl_id", "canonical_smiles"])
                data = data.merge(smiles_df, on='molecule_chembl_id')
                
                output_path = os.path.join(output_dir, output_file)
                data.to_csv(output_path, index=False)
                logger.info(f"Activity data for {chembl_id} saved to {output_path}")
            else:
                logger.info(f"No SMILES data found for {chembl_id}.")
        else:
            logger.info(f"No activity data found for {chembl_id}.")
            
    except Exception as e:
        logger.error(f"Error processing target {chembl_id}: {e}")

def download_target(args):
    """Wrapper function to run async download_target"""
    asyncio.run(download_target_async(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ChEMBL activity data and SMILES (Async Version)")
    parser.add_argument('--all_proteins', action='store_true', help="Download data for all protein targets in ChEMBL")
    parser.add_argument('--target_id', type=str, help="Target ChEMBL ID(s) to search for, comma-separated")
    parser.add_argument('--assay_type', type=str, default='B', help="Assay type(s) to search for, comma-separated")
    parser.add_argument('--pchembl_threshold_for_download', type=float, default=0, help="Threshold for pChembl value to determine active/inactive")
    parser.add_argument('--output_file', type=str, default='activity_data.csv', help="Output file to save activity data")
    parser.add_argument('--max_cores', type=int, default=multiprocessing.cpu_count() - 1, help="Maximum number of CPU cores to use")
    parser.add_argument('--smiles_input_file', type=str, help="Path to txt file containing ChEMBL IDs")
    parser.add_argument('--max_concurrent', type=int, default=50, help="Maximum number of concurrent requests")
    parser.add_argument('--target_process_batch_size', type=int, default=10, help="Number of targets to process in each batch")

    args = parser.parse_args()

    download_target(args)