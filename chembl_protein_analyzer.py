import requests
import random
import time
from typing import List, Dict, Optional
import json
from datetime import datetime, timedelta
import sys

class ChEMBLProteinAnalyzer:
    """
    ChEMBL API'den SINGLE PROTEIN tipi target proteinleri analiz eden sınıf.
    """
    
    def __init__(self, base_url: str = "https://www.ebi.ac.uk/chembl/api/data"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'ChEMBL-Protein-Analyzer/1.0'
        })
        # Connection pooling ve retry ayarları
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.timing_stats = {}
    
    def print_with_time(self, message: str, level: str = "INFO"):
        """Zamanlı ve renkli çıktı yazdırır"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        colors = {
            "INFO": "\033[94m",    # Mavi
            "SUCCESS": "\033[92m", # Yeşil  
            "WARNING": "\033[93m", # Sarı
            "ERROR": "\033[91m",   # Kırmızı
            "PROGRESS": "\033[96m" # Cyan
        }
        reset = "\033[0m"
        color = colors.get(level, colors["INFO"])
        print(f"{color}[{timestamp}] {level}: {message}{reset}")
    
    def print_progress_bar(self, current: int, total: int, prefix: str = "", width: int = 40):
        """ASCII progress bar yazdırır"""
        if total == 0:
            return
        percent = (current / total) * 100
        filled = int(width * current // total)
        bar = "█" * filled + "░" * (width - filled)
        elapsed_str = f"({current}/{total})"
        self.print_with_time(f"{prefix} |{bar}| {percent:.1f}% {elapsed_str}", "PROGRESS")
    
    def format_duration(self, seconds: float) -> str:
        """Süreyi okunabilir formata çevirir"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
    
    def get_single_protein_targets(self) -> List[Dict]:
        """
        ChEMBL API'den tüm SINGLE PROTEIN tipi target'ları alır.
        
        Returns:
            List[Dict]: Target protein listesi
        """
        start_time = time.time()
        self.print_with_time("🔍 SINGLE PROTEIN target'ları alınıyor...", "INFO")
        
        url = f"{self.base_url}/target"
        params = {
            'target_type': 'SINGLE PROTEIN',
            'format': 'json',
            'limit': 1000  # Batch size
        }
        
        all_targets = []
        offset = 0
        batch_count = 0
        
        while True:
            batch_start = time.time()
            params['offset'] = offset
            batch_count += 1
            
            try:
                self.print_with_time(f"📦 Batch #{batch_count} isteniyor... (offset: {offset})", "INFO")
                
                # Daha uzun timeout ve retry
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                
                batch_request_time = time.time() - batch_start
                self.print_with_time(f"⚡ API çağrısı tamamlandı ({self.format_duration(batch_request_time)})", "SUCCESS")
                
                data = response.json()
                targets = data.get('targets', [])
                
                if not targets:
                    self.print_with_time("✅ Tüm batch'ler tamamlandı", "SUCCESS")
                    break
                
                # Gerekli alanları filtrele
                filtered_targets = []
                filter_start = time.time()
                
                for target in targets:
                    if target.get('target_chembl_id') and target.get('pref_name'):
                        filtered_targets.append({
                            'target_chembl_id': target['target_chembl_id'],
                            'pref_name': target['pref_name']
                        })
                
                filter_time = time.time() - filter_start
                all_targets.extend(filtered_targets)
                
                self.print_with_time(f"🔧 {len(filtered_targets)} protein filtelendi ({self.format_duration(filter_time)})", "INFO")
                self.print_with_time(f"📊 Toplam biriken: {len(all_targets)} protein", "SUCCESS")
                
                # Eğer mevcut batch'te target sayısı limit'ten azsa, son batch'teyiz
                if len(targets) < params['limit']:
                    self.print_with_time("🏁 Son batch'e ulaşıldı", "INFO")
                    break
                
                offset += params['limit']
                time.sleep(1.0)  # Daha uzun API rate limiting
                
            except requests.exceptions.RequestException as e:
                self.print_with_time(f"❌ Target alınırken hata: {e}", "ERROR")
                break
        
        total_time = time.time() - start_time
        self.timing_stats['target_fetch_time'] = total_time
        
        self.print_with_time("=" * 60, "INFO")
        self.print_with_time(f"🎯 TARGET ALMA TAMAMLANDI", "SUCCESS")
        self.print_with_time(f"📈 Toplam protein sayısı: {len(all_targets)}", "SUCCESS")
        self.print_with_time(f"⏱️  Toplam süre: {self.format_duration(total_time)}", "SUCCESS")
        self.print_with_time(f"🔄 Batch sayısı: {batch_count}", "SUCCESS")
        if len(all_targets) > 0:
            avg_time = total_time / len(all_targets)
            self.print_with_time(f"⚡ Protein başına ortalama: {self.format_duration(avg_time)}", "SUCCESS")
        self.print_with_time("=" * 60, "INFO")
        
        return all_targets
    
    def count_active_compounds(self, target_chembl_id: str) -> Optional[int]:
        """
        Belirli bir target için pChEMBL >= 6 olan compound sayısını döndürür.
        
        Args:
            target_chembl_id (str): Target'ın ChEMBL ID'si
            
        Returns:
            Optional[int]: Aktif compound sayısı, hata durumunda None
        """
        request_start = time.time()
        
        url = f"{self.base_url}/activity"
        params = {
            'target_chembl_id': target_chembl_id,
            'pchembl_value__gte': 6,
            'format': 'json',
            'limit': 1  # Sadece sayıyı öğrenmek istiyoruz
        }
        
        try:
            # Daha uzun timeout ve retry mekanizması
            response = self.session.get(url, params=params, timeout=90)
            response.raise_for_status()
            
            request_time = time.time() - request_start
            
            data = response.json()
            count = data.get('page_meta', {}).get('total_count', 0)
            
            # Kısa zamanlı işlemlerde de gösterelim
            if request_time > 0.5:  # 500ms'den uzunsa göster
                self.print_with_time(f"  ⏱️  API çağrısı: {self.format_duration(request_time)}", "INFO")
            
            return count
            
        except requests.exceptions.RequestException as e:
            request_time = time.time() - request_start
            self.print_with_time(f"  ❌ {target_chembl_id} aktivite hatası ({self.format_duration(request_time)}): {e}", "ERROR")
            self.print_with_time(f"  🔄 API hatası - 5 saniye bekleniyor...", "WARNING")
            time.sleep(5.0)  # Hata sonrası daha uzun bekleme
            return None
    
    def analyze_proteins(self, sample_ratio: float = 1.0, min_compounds: int = 50) -> List[Dict]:
        """
        Protein analizi gerçekleştirir.
        
        Args:
            sample_ratio (float): Örnekleme oranı (0.0-1.0)
            min_compounds (int): Minimum compound sayısı
            
        Returns:
            List[Dict]: Kriterleiri sağlayan protein listesi
        """
        analysis_start = time.time()
        
        if not 0 <= sample_ratio <= 1:
            raise ValueError("sample_ratio 0 ile 1 arasında olmalıdır")
        
        self.print_with_time("🧬 PROTEİN ANALİZİ BAŞLANIYOR", "INFO")
        self.print_with_time(f"🎯 Hedef: >= {min_compounds} aktif compound", "INFO")
        self.print_with_time(f"📊 Örnekleme oranı: %{sample_ratio*100:.1f}", "INFO")
        
        # Tüm target'ları al
        all_targets = self.get_single_protein_targets()
        
        if not all_targets:
            self.print_with_time("❌ Hiç target protein bulunamadı.", "ERROR")
            return []
        
        # Örnekleme yap
        sampling_start = time.time()
        if sample_ratio < 1.0:
            sample_size = int(len(all_targets) * sample_ratio)
            sampled_targets = random.sample(all_targets, sample_size)
            sampling_time = time.time() - sampling_start
            self.print_with_time(f"🎲 Örnekleme tamamlandı ({self.format_duration(sampling_time)})", "SUCCESS")
            self.print_with_time(f"📋 {len(all_targets)} → {len(sampled_targets)} protein seçildi (%{sample_ratio*100:.1f})", "SUCCESS")
        else:
            sampled_targets = all_targets
            sampling_time = time.time() - sampling_start
            self.print_with_time(f"📋 Tüm {len(sampled_targets)} protein analiz edilecek", "INFO")
        
        self.timing_stats['sampling_time'] = sampling_time
        
        # Her target için aktivite sayısını kontrol et
        self.print_with_time("=" * 60, "INFO")
        self.print_with_time("🔬 AKTİVİTE ANALİZİ BAŞLANIYOR", "INFO")
        self.print_with_time("=" * 60, "INFO")
        
        qualifying_proteins = []
        total_api_calls = 0
        successful_calls = 0
        failed_calls = 0
        analysis_times = []
        
        for i, target in enumerate(sampled_targets, 1):
            target_start = time.time()
            target_id = target['target_chembl_id']
            target_name = target['pref_name']
            
            # Progress bar göster (her 10 işlemde bir)
            if i % max(1, len(sampled_targets) // 20) == 0 or i == 1:
                self.print_progress_bar(i-1, len(sampled_targets), "🔬 Analiz ilerliyor")
            
            self.print_with_time(f"[{i:3d}/{len(sampled_targets)}] 🧪 {target_id}", "INFO")
            self.print_with_time(f"         📝 {target_name[:60]}{'...' if len(target_name) > 60 else ''}", "INFO")
            
            total_api_calls += 1
            compound_count = self.count_active_compounds(target_id)
            
            target_time = time.time() - target_start
            analysis_times.append(target_time)
            
            if compound_count is not None:
                successful_calls += 1
                self.print_with_time(f"         📊 {compound_count} aktif compound bulundu", "SUCCESS")
                
                if compound_count >= min_compounds:
                    protein_info = {
                        'target_chembl_id': target_id,
                        'pref_name': target_name,
                        'active_compound_count': compound_count
                    }
                    qualifying_proteins.append(protein_info)
                    self.print_with_time(f"         ✅ KRİTERLERİ KARŞILIYOR! (>= {min_compounds})", "SUCCESS")
                else:
                    self.print_with_time(f"         ❌ Yetersiz ({compound_count} < {min_compounds})", "WARNING")
            else:
                failed_calls += 1
                self.print_with_time(f"         ❌ Aktivite verisi alınamadı", "ERROR")
            
            # Timing bilgisi
            self.print_with_time(f"         ⏱️  İşlem süresi: {self.format_duration(target_time)}", "INFO")
            
            # ETA hesaplama
            if i > 0:
                avg_time = sum(analysis_times) / len(analysis_times)
                remaining = len(sampled_targets) - i
                eta = avg_time * remaining
                self.print_with_time(f"         🕐 Tahmini kalan süre: {self.format_duration(eta)}", "INFO")
            
            self.print_with_time("", "INFO")  # Boş satır
            
            # Daha uzun API rate limiting (ChEMBL sunucusu yoğun)
            time.sleep(2.0)
        
        # Son progress bar
        self.print_progress_bar(len(sampled_targets), len(sampled_targets), "🔬 Analiz tamamlandı")
        
        # Genel istatistikler
        total_analysis_time = time.time() - analysis_start
        self.timing_stats['total_analysis_time'] = total_analysis_time
        self.timing_stats['activity_analysis_time'] = total_analysis_time - self.timing_stats.get('target_fetch_time', 0) - sampling_time
        
        self.print_timing_summary(
            total_analysis_time, total_api_calls, successful_calls, 
            failed_calls, len(qualifying_proteins), analysis_times
        )
        
        return qualifying_proteins
    
    def print_timing_summary(self, total_time: float, total_calls: int, 
                           successful_calls: int, failed_calls: int, 
                           qualifying_count: int, analysis_times: List[float]):
        """Detaylı zamanlama özetini yazdırır"""
        self.print_with_time("=" * 60, "INFO")
        self.print_with_time("📈 DETAYLI PERFORMANS RAPORU", "SUCCESS")
        self.print_with_time("=" * 60, "INFO")
        
        # Genel zamanlar
        target_time = self.timing_stats.get('target_fetch_time', 0)
        sampling_time = self.timing_stats.get('sampling_time', 0)
        activity_time = self.timing_stats.get('activity_analysis_time', 0)
        
        self.print_with_time(f"🕐 TOPLAM SÜRE: {self.format_duration(total_time)}", "SUCCESS")
        self.print_with_time(f"  ├─ Target alma: {self.format_duration(target_time)} ({target_time/total_time*100:.1f}%)", "INFO")
        self.print_with_time(f"  ├─ Örnekleme: {self.format_duration(sampling_time)} ({sampling_time/total_time*100:.1f}%)", "INFO")
        self.print_with_time(f"  └─ Aktivite analizi: {self.format_duration(activity_time)} ({activity_time/total_time*100:.1f}%)", "INFO")
        
        # API performansı
        self.print_with_time("", "INFO")
        self.print_with_time("🌐 API PERFORMANSI:", "SUCCESS")
        self.print_with_time(f"  ├─ Toplam API çağrısı: {total_calls}", "INFO")
        self.print_with_time(f"  ├─ Başarılı: {successful_calls} ({successful_calls/total_calls*100:.1f}%)", "SUCCESS")
        self.print_with_time(f"  └─ Başarısız: {failed_calls} ({failed_calls/total_calls*100:.1f}%)", "WARNING" if failed_calls > 0 else "SUCCESS")
        
        if analysis_times:
            avg_analysis = sum(analysis_times) / len(analysis_times)
            min_analysis = min(analysis_times)
            max_analysis = max(analysis_times)
            
            self.print_with_time("", "INFO")
            self.print_with_time("⚡ ANALİZ SÜRE İSTATİSTİKLERİ:", "SUCCESS")
            self.print_with_time(f"  ├─ Ortalama: {self.format_duration(avg_analysis)}", "INFO")
            self.print_with_time(f"  ├─ En hızlı: {self.format_duration(min_analysis)}", "SUCCESS")
            self.print_with_time(f"  └─ En yavaş: {self.format_duration(max_analysis)}", "WARNING")
        
        # Sonuç özeti
        self.print_with_time("", "INFO")
        self.print_with_time("🎯 SONUÇ ÖZETİ:", "SUCCESS")
        self.print_with_time(f"  ├─ Analiz edilen protein: {total_calls}", "INFO")
        self.print_with_time(f"  ├─ Kriterleri karşılayan: {qualifying_count}", "SUCCESS")
        self.print_with_time(f"  └─ Başarı oranı: {qualifying_count/total_calls*100:.1f}%", "SUCCESS")
        
        # Performans metrikleri
        if total_time > 0:
            proteins_per_sec = total_calls / total_time
            self.print_with_time("", "INFO")
            self.print_with_time("📊 PERFORMANS METRİKLERİ:", "SUCCESS")
            self.print_with_time(f"  ├─ Protein/saniye: {proteins_per_sec:.2f}", "INFO")
            self.print_with_time(f"  └─ Saniye/protein: {total_time/total_calls:.2f}", "INFO")
        
        self.print_with_time("=" * 60, "INFO")
    
    def save_results(self, results: List[Dict], filename: str = "chembl_qualifying_proteins.json"):
        """
        Sonuçları JSON dosyasına kaydeder.
        
        Args:
            results (List[Dict]): Sonuç listesi
            filename (str): Kaydedilecek dosya adı
        """
        save_start = time.time()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            save_time = time.time() - save_start
            file_size = len(json.dumps(results, indent=2, ensure_ascii=False).encode('utf-8'))
            
            self.print_with_time("", "INFO")
            self.print_with_time("💾 DOSYA KAYDETME", "SUCCESS")
            self.print_with_time(f"  ├─ Dosya: {filename}", "INFO")
            self.print_with_time(f"  ├─ Boyut: {file_size:,} byte", "INFO")
            self.print_with_time(f"  ├─ Kayıt sayısı: {len(results)}", "INFO")
            self.print_with_time(f"  └─ Süre: {self.format_duration(save_time)}", "INFO")
            
        except Exception as e:
            save_time = time.time() - save_start
            self.print_with_time(f"❌ Kaydetme hatası ({self.format_duration(save_time)}): {e}", "ERROR")


def main():
    """
    Ana fonksiyon - kullanım örneği
    """
    main_start = time.time()
    
    analyzer = ChEMBLProteinAnalyzer()
    
    # Parametreler
    sample_ratio = 0.1  # %10 örnekleme (hızlı test için)
    min_compounds = 50  # Minimum 50 aktif compound
    
    analyzer.print_with_time("🚀 ChEMBL Protein Analyzer BAŞLANIYOR", "SUCCESS")
    analyzer.print_with_time("=" * 60, "INFO")
    analyzer.print_with_time("📋 PARAMETRELER:", "INFO")
    analyzer.print_with_time(f"  ├─ Örnekleme oranı: %{sample_ratio*100}", "INFO")
    analyzer.print_with_time(f"  ├─ Minimum compound: {min_compounds}", "INFO")
    analyzer.print_with_time(f"  └─ pChEMBL kriteri: >= 6", "INFO")
    analyzer.print_with_time("=" * 60, "INFO")
    
    try:
        # Analizi çalıştır
        qualifying_proteins = analyzer.analyze_proteins(
            sample_ratio=sample_ratio,
            min_compounds=min_compounds
        )
        
        # Sonuçları göster
        analyzer.print_with_time("", "INFO")
        analyzer.print_with_time("🎯 SONUÇ DETAYLARI", "SUCCESS")
        analyzer.print_with_time("=" * 60, "INFO")
        
        if qualifying_proteins:
            analyzer.print_with_time(f"✅ {len(qualifying_proteins)} protein kriterleri karşılıyor:", "SUCCESS")
            analyzer.print_with_time("", "INFO")
            
            # En yüksek compound sayılı proteinleri önce göster
            sorted_proteins = sorted(qualifying_proteins, 
                                   key=lambda x: x['active_compound_count'], 
                                   reverse=True)
            
            for i, protein in enumerate(sorted_proteins, 1):
                analyzer.print_with_time(f"🏆 {i:2d}. {protein['target_chembl_id']}", "SUCCESS")
                analyzer.print_with_time(f"     📝 {protein['pref_name']}", "INFO")
                analyzer.print_with_time(f"     💊 {protein['active_compound_count']:,} aktif compound", "SUCCESS")
                analyzer.print_with_time("", "INFO")
            
            # Sonuçları kaydet
            analyzer.save_results(qualifying_proteins)
            
            # İstatistiksel özet
            compound_counts = [p['active_compound_count'] for p in qualifying_proteins]
            analyzer.print_with_time("", "INFO")
            analyzer.print_with_time("📊 İSTATİSTİKSEL ÖZET:", "SUCCESS")
            analyzer.print_with_time(f"  ├─ Ortalama compound: {sum(compound_counts)/len(compound_counts):.1f}", "INFO")
            analyzer.print_with_time(f"  ├─ En yüksek: {max(compound_counts):,}", "SUCCESS")
            analyzer.print_with_time(f"  └─ En düşük: {min(compound_counts):,}", "INFO")
            
        else:
            analyzer.print_with_time("❌ Kriterleri karşılayan protein bulunamadı.", "WARNING")
    
    except KeyboardInterrupt:
        analyzer.print_with_time("⚠️  İşlem kullanıcı tarafından durduruldu.", "WARNING")
    except Exception as e:
        analyzer.print_with_time(f"❌ Analiz sırasında hata oluştu: {e}", "ERROR")
    
    finally:
        # Toplam çalışma süresi
        total_main_time = time.time() - main_start
        analyzer.print_with_time("", "INFO")
        analyzer.print_with_time("🏁 PROGRAM TAMAMLANDI", "SUCCESS")
        analyzer.print_with_time(f"🕐 Toplam çalışma süresi: {analyzer.format_duration(total_main_time)}", "SUCCESS")
        analyzer.print_with_time("=" * 60, "INFO")


if __name__ == "__main__":
    main() 