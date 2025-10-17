from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Vamos a procesar el texto cargado manualmente
file_path = "./results_yep.txt"

hyps = []
clean_preds = []
noisy_preds = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if "HYPS:" in line and "CLEAN PRED:" in line and "NOISY:" in line:
            # Extraer referencias (HYPS)
            start_h = line.find("HYPS:") + len("HYPS:")
            end_h = line.find("NOISY:")
            hyps_text = line[start_h:end_h].strip()
            hyps_list = eval(hyps_text) if hyps_text.startswith("[") else [hyps_text]

            # Extraer NOISY
            start_n = line.find("NOISY:") + len("NOISY:")
            end_n = line.find("PRED:")
            noisy = line[start_n:end_n].strip()

            # Extraer CLEAN PRED
            start_c = line.find("CLEAN PRED:") + len("CLEAN PRED:")
            clean = line[start_c:].strip()
            
            if hyps_list and clean and noisy:
                hyps.append([h.split() for h in hyps_list])
                clean_preds.append(clean.split())
                noisy_preds.append(noisy.split())
            
            print(hyps_list)
            if len(hyps) > 1:
                print(f"HYPS: {hyps}")
                exit()

# Calcular BLEU
smooth = SmoothingFunction().method1
bleu_clean = corpus_bleu(hyps, clean_preds, smoothing_function=smooth)
bleu_noisy = corpus_bleu(hyps, noisy_preds, smoothing_function=smooth)

bleu_clean, bleu_noisy
