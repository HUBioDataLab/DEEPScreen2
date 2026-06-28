#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Molekül Görüntüsü Üretici — Saydam Arka Plan
=============================================
Verilen SMILES listesinden 300x300 saydam arka planlı PNG üretir.
Çıktı: ./mol_images/<molecule_id>.png

Kullanım:
    python generate_mol_images.py --input molecules.csv --output_dir ./mol_images
    
CSV formatı (zorunlu sütunlar):
    molecule_id, smiles
    CHEMBL100675, CCOc1ccc(...)cc1
"""

import os
import argparse
import numpy as np
import pandas as pd
from io import BytesIO

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("⚠️  RDKit bulunamadı. 'pip install rdkit' ile kurun.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  Pillow bulunamadı. 'pip install Pillow' ile kurun.")


def smiles_to_transparent_png(smiles: str, size: int = 300) -> "Image":
    """
    SMILES'tan saydam arka planlı PIL Image üretir.
    Renk bazlı ayırt etme: N (mavi), O (kırmızı), Br (kahve) gibi atom etiketleri
    tam opak korunur; beyaz/gri arka plan kademeli olarak saydam yapılır.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Geçersiz SMILES: {smiles}")

    # RDKit SVG/PNG renderer — beyaz arka plan
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.drawOptions().clearBackground = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png_bytes = drawer.GetDrawingText()
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")

    data = np.array(img, dtype=float)
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    # Parlaklık: pikselin ne kadar açık olduğu
    brightness = (r + g + b) / 3.0

    # Renklililik: R/G/B kanalları arasındaki maksimum fark
    # Yüksekse renkli piksel (N, O, Br atom etiketleri), düşükse gri/beyaz (arka plan)
    colorfulness = (np.max(data[:, :, :3], axis=2) -
                    np.min(data[:, :, :3], axis=2))

    # Arka plan maskesi: hem açık (brightness > 200) hem renksiz (colorfulness < 15)
    is_background = (brightness > 200) & (colorfulness < 15)

    # Alpha hesaplama:
    # - Arka plan: brightness'a göre kademeli saydamlık (anti-aliasing kenarları için)
    # - Renkli/koyu pikseller: tam opak (255)
    background_alpha = np.clip((255.0 - brightness) * 3.0, 0, 255)
    alpha_channel = np.where(is_background, background_alpha, 255.0)

    data[:, :, 3] = alpha_channel
    return Image.fromarray(data.astype(np.uint8))


def generate_from_csv(input_csv: str, output_dir: str, size: int = 300):
    """CSV dosyasından toplu görüntü üretir."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    required = {'molecule_id', 'smiles'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV'de şu sütunlar olmalı: {required}. Bulunanlar: {list(df.columns)}")

    success, fail = 0, 0
    for _, row in df.iterrows():
        mol_id = str(row['molecule_id'])
        smiles = str(row['smiles'])
        out_path = os.path.join(output_dir, f"{mol_id}.png")

        try:
            img = smiles_to_transparent_png(smiles, size=size)
            img.save(out_path, format="PNG")
            print(f"✓ {mol_id} → {out_path}")
            success += 1
        except Exception as e:
            print(f"✗ {mol_id}: {e}")
            fail += 1

    print(f"\n✅ Tamamlandı: {success} başarılı, {fail} başarısız.")


def generate_from_dict(molecules: dict, output_dir: str, size: int = 300):
    """
    Dict'ten toplu görüntü üretir.
    molecules = {"CHEMBL100675": "CCO...", "DILI1": "c1ccc..."}
    """
    os.makedirs(output_dir, exist_ok=True)
    success, fail = 0, 0

    for mol_id, smiles in molecules.items():
        out_path = os.path.join(output_dir, f"{mol_id}.png")
        try:
            img = smiles_to_transparent_png(smiles, size=size)
            img.save(out_path, format="PNG")
            print(f"✓ {mol_id} → {out_path}")
            success += 1
        except Exception as e:
            print(f"✗ {mol_id}: {e}")
            fail += 1

    print(f"\n✅ Tamamlandı: {success} başarılı, {fail} başarısız.")


def main():
    parser = argparse.ArgumentParser(description="Saydam arka planlı molekül PNG üretici")
    parser.add_argument("--input", type=str, required=True,
                        help="CSV dosyası (sütunlar: molecule_id, smiles)")
    parser.add_argument("--output_dir", type=str, default="./mol_images",
                        help="Çıktı klasörü (varsayılan: ./mol_images)")
    parser.add_argument("--size", type=int, default=300,
                        help="Görüntü boyutu piksel (varsayılan: 300)")
    args = parser.parse_args()

    if not HAS_RDKIT or not HAS_PIL:
        print("Eksik bağımlılık. Çıkılıyor.")
        return

    generate_from_csv(args.input, args.output_dir, size=args.size)


if __name__ == "__main__":
    main()