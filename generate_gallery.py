#!/usr/bin/env python3
"""
Generate HTML gallery for PCB inference results.
"""
import os
import json
from pathlib import Path

output_dir = "/home/xinrui/projects/OminiControl/inference_output_pcb"
manifest_path = "/home/xinrui/projects/data/ti_pcb/COCO_label/cropped_512/composite_manifest_test.json"

# Load manifest to get prompts
with open(manifest_path) as f:
    samples = json.load(f)

# Get all comparison images
comparison_files = sorted([f for f in os.listdir(output_dir) if f.endswith("_comparison.jpg")])

html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OminiControl PCB Harmonization - Step 18000 Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            padding: 20px;
        }
        
        .header {
            max-width: 1400px;
            margin: 0 auto 40px;
            padding: 30px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .info-item {
            padding: 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }
        
        .info-item strong {
            color: #667eea;
            display: block;
            margin-bottom: 5px;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .legend {
            max-width: 1400px;
            margin: 0 auto 30px;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            display: flex;
            gap: 30px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .legend-box {
            width: 60px;
            height: 40px;
            border-radius: 4px;
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .legend-box.composite { background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%); }
        .legend-box.generated { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .legend-box.real { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
        
        .gallery {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 30px;
        }
        
        .sample {
            background: #1a1a1a;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .sample:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        .sample img {
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }
        
        .sample-info {
            padding: 15px;
            background: rgba(255,255,255,0.02);
        }
        
        .sample-title {
            font-weight: 600;
            font-size: 1.1em;
            margin-bottom: 8px;
            color: #667eea;
        }
        
        .sample-prompt {
            font-size: 0.9em;
            color: #a0a0a0;
            line-height: 1.4;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            padding: 20px;
        }
        
        .modal-content {
            max-width: 90%;
            max-height: 90%;
            margin: auto;
            display: block;
            position: relative;
            top: 50%;
            transform: translateY(-50%);
        }
        
        .close {
            position: absolute;
            top: 30px;
            right: 40px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
        }
        
        .close:hover {
            color: #667eea;
        }
        
        @media (max-width: 768px) {
            .gallery {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 OminiControl PCB Harmonization Results</h1>
        <p style="font-size: 1.1em; margin-top: 10px; color: #b0b0b0;">
            Checkpoint: <strong style="color: #fff;">step_18000</strong> | 
            Model: <strong style="color: #fff;">FLUX.1-dev + LoRA</strong>
        </p>
        
        <div class="info">
            <div class="info-item">
                <strong>Dataset</strong>
                TI PCB Test Set
            </div>
            <div class="info-item">
                <strong>Samples</strong>
                """ + str(len(comparison_files)) + """ images
            </div>
            <div class="info-item">
                <strong>Resolution</strong>
                512 × 512
            </div>
            <div class="info-item">
                <strong>Steps</strong>
                28 diffusion steps
            </div>
            <div class="info-item">
                <strong>Task</strong>
                Component composite → realistic PCB
            </div>
        </div>
    </div>
    
    <div class="legend">
        <strong style="color: #fff;">Image Layout:</strong>
        <div class="legend-item">
            <div class="legend-box composite"></div>
            <span>Composite Input</span>
        </div>
        <div class="legend-item">
            <div class="legend-box generated"></div>
            <span>Generated Output</span>
        </div>
        <div class="legend-item">
            <div class="legend-box real"></div>
            <span>Ground Truth</span>
        </div>
    </div>
    
    <div class="gallery">
"""

# Add each sample
for i, img_file in enumerate(comparison_files):
    sample_idx = int(img_file.split("_")[1])
    if sample_idx < len(samples):
        sample = samples[sample_idx]
        prompt = sample.get("prompt", "PCB board")
        name = sample.get("name", f"sample_{sample_idx}")
    else:
        prompt = "PCB board"
        name = f"sample_{sample_idx}"
    
    html += f"""
        <div class="sample">
            <img src="{img_file}" alt="Sample {sample_idx}" onclick="openModal(this.src)">
            <div class="sample-info">
                <div class="sample-title">Sample {sample_idx:03d}: {name}</div>
                <div class="sample-prompt">{prompt}</div>
            </div>
        </div>
"""

html += """
    </div>
    
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        function openModal(src) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""

# Write HTML file
output_path = os.path.join(output_dir, "results_gallery.html")
with open(output_path, 'w') as f:
    f.write(html)

print(f"Gallery created: {output_path}")
print(f"Total samples: {len(comparison_files)}")
