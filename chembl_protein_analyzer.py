#!/usr/bin/env python3
"""
Fast ChEMBL Protein Analysis System
===================================

This script provides a high-performance parallel analysis tool for ChEMBL protein targets.
It can analyze thousands of protein targets efficiently using concurrent requests.

Features:
- Parallel processing with configurable worker threads
- Rate limiting to respect API limits
- Session pooling for optimal connection reuse
- Comprehensive error handling and retry mechanisms
- Statistical analysis and result export

Usage:
    python simple_chembl_analyzer_v3fast.py --ratio 0.01 --min-compounds 50 --workers 10

Arguments:
    --ratio          Sampling ratio (0.001-1.0, default: 0.001)
    --min-compounds  Minimum active compound count (default: 50)
    --workers        Number of parallel workers (default: 10)
    --rate           Requests per second (default: 5)

Output:
    - JSON files with all analyzed proteins and qualifying proteins
    - Statistical summary of the analysis
    - Performance metrics and timing information

Author: ChEMBL Analysis Team
Version: 3.0 (Fast Parallel Edition)
"""

import requests
import random
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

class FastChEMBLAnalyzer:
    """
    Parallel ChEMBL analysis tool - 5-10x faster than sequential processing
    """
    
    def __init__(self, max_workers: int = 10, requests_per_second: int = 5):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.max_workers = max_workers
        self.requests_per_second = requests_per_second
        self.delay_between_requests = 1.0 / requests_per_second
        
        # For rate limiting
        self.last_request_time = {}
        self.lock = threading.Lock()
        
        # Session pool - separate session for each thread
        self.session_pool = Queue()
        for _ in range(max_workers):
            session = self._create_session()
            self.session_pool.put(session)
    
    def _create_session(self):
        """Creates a separate session for each thread"""
        session = requests.Session()
        session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'ChEMBL-Fast-Analyzer/2.0'
        })
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit(self):
        """Thread-safe rate limiting"""
        with self.lock:
            thread_id = threading.current_thread().ident
            current_time = time.time()
            
            if thread_id in self.last_request_time:
                elapsed = current_time - self.last_request_time[thread_id]
                if elapsed < self.delay_between_requests:
                    time.sleep(self.delay_between_requests - elapsed)
            
            self.last_request_time[thread_id] = time.time()
    
    def get_total_protein_count(self) -> int:
        """Gets the total number of SINGLE PROTEIN targets"""
        print("Checking total SINGLE PROTEIN count in ChEMBL...")
        
        session = self._create_session()
        url = f"{self.base_url}/target"
        params = {
            'target_type': 'SINGLE PROTEIN',
            'format': 'json',
            'limit': 1
        }
        
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            total_count = data.get('page_meta', {}).get('total_count', 0)
            
            print(f"Found {total_count:,} SINGLE PROTEIN targets in ChEMBL")
            return total_count
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting total count: {e}")
            return 0
        finally:
            session.close()

    def get_protein_targets(self) -> List[Dict]:
        """Retrieves SINGLE PROTEIN targets"""
        print("Retrieving target proteins...")
        
        session = self._create_session()
        url = f"{self.base_url}/target"
        params = {
            'target_type': 'SINGLE PROTEIN',
            'format': 'json',
            'limit': 1000
        }
        
        all_targets = []
        offset = 0
        
        try:
            while True:
                params['offset'] = offset
                
                response = session.get(url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                targets = data.get('targets', [])
                
                if not targets:
                    break
                
                for target in targets:
                    if target.get('target_id') and target.get('pref_name'):
                        all_targets.append({
                            'target_id': target['target_id'],
                            'pref_name': target['pref_name']
                        })
                
                print(f"  {len(all_targets)} proteins collected...")
                
                if len(targets) < params['limit']:
                    break
                
                offset += params['limit']
                time.sleep(0.5)  # Reduced delay
                
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
        finally:
            session.close()
        
        print(f"Total {len(all_targets)} proteins found")
        return all_targets
    
    def get_active_compound_count_worker(self, target: Dict) -> Dict:
        """Worker function for parallel processing"""
        # Get session from session pool
        session = self.session_pool.get()
        
        try:
            target_id = target['target_id']
            target_name = target['pref_name']
            
            # Rate limiting
            self._rate_limit()
            
            url = f"{self.base_url}/activity"
            params = {
                'target_id': target_id,
                'pchembl_value__gte': 6,
                'format': 'json',
                'limit': 1
            }
            
            response = session.get(url, params=params, timeout=90)
            response.raise_for_status()
            
            data = response.json()
            compound_count = data.get('page_meta', {}).get('total_count', 0)
            
            return {
                'target_id': target_id,
                'pref_name': target_name,
                'active_compound_count': compound_count,
                'error': False
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'target_id': target.get('target_id', 'unknown'),
                'pref_name': target.get('pref_name', 'unknown'),
                'active_compound_count': None,
                'error': True,
                'error_message': str(e)
            }
        finally:
            # Return session to pool
            self.session_pool.put(session)

    def analyze_parallel(self, sample_ratio: float = 0.001, min_compounds: int = 50):
        """Parallel analysis - Main speedup function"""
        print(f"\nFast ChEMBL Protein Analysis")
        print(f"Sampling: {sample_ratio*100}%")
        print(f"Minimum compounds: {min_compounds}")
        print(f"Workers: {self.max_workers}")
        print(f"Rate: {self.requests_per_second} req/sec")
        print("=" * 50)
        
        # Get targets
        all_targets = self.get_protein_targets()
        if not all_targets:
            print("No proteins found")
            return
        
        # Sample
        sample_size = int(len(all_targets) * sample_ratio)
        sampled_targets = random.sample(all_targets, sample_size)
        print(f"\n{len(all_targets)} -> {len(sampled_targets)} proteins selected")
        
        # Parallel analysis
        print(f"\nAnalyzing {len(sampled_targets)} proteins in parallel...")
        
        all_analyzed_proteins = []
        qualifying_proteins = []
        start_time = time.time()
        completed = 0
        
        # ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_target = {
                executor.submit(self.get_active_compound_count_worker, target): target 
                for target in sampled_targets
            }
            
            # Collect results
            for future in as_completed(future_to_target):
                result = future.result()
                completed += 1
                
                # Process result
                target_id = result['target_id']
                compound_count = result['active_compound_count']
                
                if not result['error'] and compound_count is not None:
                    status = "PASS" if compound_count >= min_compounds else "FAIL"
                    print(f"[{completed:3d}/{len(sampled_targets)}] {target_id}: {compound_count:,} compounds {status}")
                    
                    protein_data = {
                        'target_id': result['target_id'],
                        'pref_name': result['pref_name'],
                        'active_compound_count': compound_count,
                        'meets_criteria': compound_count >= min_compounds
                    }
                    
                    if compound_count >= min_compounds:
                        qualifying_proteins.append(protein_data)
                    
                else:
                    print(f"[{completed:3d}/{len(sampled_targets)}] {target_id}: ERROR")
                    protein_data = {
                        'target_id': result['target_id'],
                        'pref_name': result['pref_name'],
                        'active_compound_count': None,
                        'meets_criteria': False,
                        'error': True
                    }
                
                all_analyzed_proteins.append(protein_data)
                
                # Show progress
                if completed % 10 == 0 or completed == len(sampled_targets):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = len(sampled_targets) - completed
                    eta = remaining / rate if rate > 0 else 0
                    
                    print(f"    Progress: {(completed/len(sampled_targets)*100):.1f}% | "
                          f"Rate: {rate:.1f}/sec | ETA: {eta:.0f}s")
        
        # Results
        total_time = time.time() - start_time
        self._show_results(all_analyzed_proteins, qualifying_proteins, 
                          total_time, sample_ratio, min_compounds, sampled_targets)

    def _show_results(self, all_analyzed_proteins, qualifying_proteins, 
                     total_time, sample_ratio, min_compounds, sampled_targets):
        """Show and save results"""
        
        print(f"\nAnalysis time: {total_time/60:.1f} minutes")
        print(f"Speed improvement: ~{(len(sampled_targets) * 2.0 / total_time):.1f}x faster!")
        print("=" * 50)
        print("RESULTS")
        print("=" * 50)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        all_filename = f"fast_all_analyzed_proteins_{sample_ratio:.3f}_{min_compounds}_{timestamp}.json"
        with open(all_filename, 'w', encoding='utf-8') as f:
            json.dump(all_analyzed_proteins, f, indent=2, ensure_ascii=False)
        print(f"All analyzed proteins saved to '{all_filename}'")
        
        if qualifying_proteins:
            # Sort by compound count
            qualifying_proteins.sort(key=lambda x: x['active_compound_count'], reverse=True)
            
            print(f"{len(qualifying_proteins)} proteins meet the criteria:\n")
            
            for i, protein in enumerate(qualifying_proteins[:10], 1):  # Show top 10
                print(f"{i:2d}. {protein['target_id']} - {protein['active_compound_count']:,} compounds")
                print(f"    {protein['pref_name'][:80]}{'...' if len(protein['pref_name']) > 80 else ''}")
                print()
            
            if len(qualifying_proteins) > 10:
                print(f"    ... and {len(qualifying_proteins) - 10} more proteins\n")
            
            # Save qualifying proteins
            qualifying_filename = f"fast_qualifying_proteins_{sample_ratio:.3f}_{min_compounds}_{timestamp}.json"
            with open(qualifying_filename, 'w', encoding='utf-8') as f:
                json.dump(qualifying_proteins, f, indent=2, ensure_ascii=False)
            
            print(f"Qualifying proteins saved to '{qualifying_filename}'")
            
            # Statistics
            compound_counts = [p['active_compound_count'] for p in qualifying_proteins]
            print(f"\nStatistics:")
            print(f"  • Average: {sum(compound_counts)/len(compound_counts):.0f} compounds")
            print(f"  • Highest: {max(compound_counts):,} compounds")
            print(f"  • Lowest: {min(compound_counts):,} compounds")
            print(f"  • Success rate: {len(qualifying_proteins)/len(all_analyzed_proteins)*100:.1f}%")
            
        else:
            print("No proteins found meeting the criteria")
        
        # Summary
        successful = len([p for p in all_analyzed_proteins if not p.get('error', False)])
        errors = len([p for p in all_analyzed_proteins if p.get('error', False)])
        
        print(f"\nAnalysis Summary:")
        print(f"  • Total proteins analyzed: {len(all_analyzed_proteins)}")
        print(f"  • Successful analyses: {successful}")
        print(f"  • Errors: {errors}")
        print(f"  • Proteins meeting criteria: {len(qualifying_proteins)}")
        print(f"  • Average analysis time per protein: {total_time/len(all_analyzed_proteins):.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Fast ChEMBL Protein Analysis System')
    parser.add_argument('--ratio', type=float, default=0.001,
                       help='Sampling ratio (0.001-1.0, default: 0.001)')
    parser.add_argument('--min-compounds', type=int, default=50,
                       help='Minimum active compound count (default: 50)')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    parser.add_argument('--rate', type=int, default=5,
                       help='Requests per second (default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.001 <= args.ratio <= 1.0):
        print("Error: ratio must be between 0.001 and 1.0")
        return
    
    if args.min_compounds < 1:
        print("Error: min-compounds must be at least 1")
        return
        
    if args.workers < 1 or args.workers > 50:
        print("Error: workers must be between 1 and 50")
        return
        
    if args.rate < 1 or args.rate > 20:
        print("Error: rate must be between 1 and 20 req/sec")
        return
    
    print("Fast ChEMBL Protein Analysis System")
    print("=" * 50)
    
    analyzer = FastChEMBLAnalyzer(max_workers=args.workers, 
                                 requests_per_second=args.rate)
    
    # Get total count
    total_count = analyzer.get_total_protein_count()
    
    if total_count == 0:
        print("Could not retrieve protein count")
        return
    
    estimated_proteins = int(total_count * args.ratio)
    estimated_time_old = estimated_proteins * 2.0 / 60  # Old method
    estimated_time_new = estimated_proteins / args.rate / 60  # New method
    
    print(f"\nPerformance Comparison:")
    print(f"  • Old method time: ~{estimated_time_old:.1f} minutes")
    print(f"  • New method time: ~{estimated_time_new:.1f} minutes")
    print(f"  • Speed improvement: ~{estimated_time_old/estimated_time_new:.1f}x faster!")
    
    print(f"\nSelected Parameters:")
    print(f"  • Sampling ratio: {args.ratio*100}%")
    print(f"  • Minimum compounds: {args.min_compounds}")
    print(f"  • Proteins to analyze: ~{estimated_proteins:,}")
    print(f"  • Parallel workers: {args.workers}")
    print(f"  • Request rate: {args.rate}/sec")
    
    try:
        confirm = input(f"\nStart fast analysis? (y/N): ")
        if confirm.lower() != 'y':
            print("Analysis cancelled")
            return
        
        analyzer.analyze_parallel(sample_ratio=args.ratio, min_compounds=args.min_compounds)
        
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()