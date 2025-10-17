import os
import torch
import gc
from tqdm import tqdm
import torch.nn.functional as F
import h5py
import numpy as np
from typing import Optional, List
from ..dataloader import KeypointDataset, collate_fn
from torch.utils.data import DataLoader
from ..utils.setup_train import  build_model, setup_paths, BatchSampler
from ..utils.config_loader import cfg
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import cast
from collections import defaultdict
from ..inference.predictor import MultimodalSignLM

device = "cuda" if torch.cuda.is_available() else "cpu"

HYPS = [
  ["a tierra", "a tierra"],
  ["abecedario", "abecedario.mp4"],
  ["abrir", "abrir"],
  ["abrir cortina", "abrir-cortina"],
  ["aburrido", "aburrido"],
  ["aceptar", "aceptar", "aceptaron", "aceptaron"],
  ["acercarse", "acercarse"],
  ["acuerdos", "acuerdos"],
  ["adivina", "adivina"],
  ["agua", "agua"],
  ["ah!", "aaah!"],
  ["ahi", "ahi", "ahí"],
  ["ahora", "ahora"],
  ["aire viento", "aire/ viento"],
  ["algunos", "algunos"],
  ["alla", "alla", "allá"],
  ["alo", "aló"],
  ["alto", "alto"],
  ["amanecer", "amanecer"],
  ["amargo", "amargo"],
  ["amarillo", "amarillo"],
  ["ambos", "ambos"],
  ["ambulancia", "ambulancia"],
  ["amigo", "amigo"],
  ["anillo", "anillo"],
  ["anochecer", "anochecer"],
  ["antes", "antes"],
  ["antigua", "antigua"],
  ["apaga", "apaga"],
  ["apellido", "apellido"],
  ["aprender", "aprender"],
  ["argentina", "argentina"],
  ["arrodillarse postrarse", "arrodillarse/ postrarse"],
  ["arroz", "arroz"],
  ["atardecer", "atardecer"],
  ["atrapar", "atrapar"],
  ["averiguar", "averiguar"],
  ["avion", "avion"],
  ["avioneta", "avioneta"],
  ["ayuda", "ayuda"],
  ["ayudar", "ayudar"],
  ["azul claro", "azul claro"],
  ["bailar", "bailar"],
  ["bajar escalera", "bajar-escalera"],
  ["banar", "bañar"],
  ["barbacoa", "barbacoa"],
  ["barco", "barco"],
  ["bastante", "bastante"],
  ["bicicleta", "bicicleta"],
  ["bien", "bien"],
  ["bola", "bola"],
  ["bola de cristal", "bola-de-cristal"],
  ["bote", "bote"],
  ["brillante", "brillante"],
  ["burlarse", "burlarse"],
  ["bus", "bus"],
  ["busca", "busca", "buscar", "buscar"],
  ["c i l m a", "c-i-l-m-a"],
  ["c l", "c-l"],
  ["caer", "caer"],
  ["caja", "caja"],
  ["cajon", "cajón"],
  ["callada", "callada"],
  ["calor", "calor"],
  ["cama", "cama"],
  ["cambiarse de ropa", "cambiarse de ropa"],
  ["caminar", "caminar"],
  ["camion", "camion"],
  ["camioneta", "camioneta"],
  ["campamento", "campamento"],
  ["campamento carpa iman", "campamento/ carpa/ imán "],
  ["captar", "captar"],
  ["cara", "cara"],
  ["caramelo", "caramelo"],
  ["cargar", "cargar"],
  ["carro", "carro"],
  ["cartas de tarot", "cartas-de-tarot"],
  ["casa", "casa"],
  ["casaca", "casaca"],
  ["casar", "casar"],
  ["catorce", "catorce"],
  ["cerrar", "cerrar"],
  ["cerrar cajon", "cerrar-cajon"],
  ["cerrar cortina", "cerrar cortina"],
  ["chau", "chau"],
  ["chocar", "chocar"],
  ["cien", "cien"],
  ["cincuenta", "cincuenta"],
  ["ciudad", "ciudad"],
  ["colores", "colores"],
  ["combi", "combi"],
  ["comer", "comer"],
  ["comprar", "comprar"],
  ["contactar", "contactar"],
  ["contar", "contar"],
  ["contar dinero", "contar-dinero"],
  ["contar numeros", "contar-numeros"],
  ["convive o se junto", "convive ó se juntó"],
  ["conyuge", "conyuge"],
  ["copiar", "copiar"],
  ["cornudo", "cornudo"],
  ["correr", "correr"],
  ["cortar", "cortar "],
  ["cortar interrumpida", "cortar/ interrumpida"],
  ["cortina", "cortina"],
  ["cortina abierta", "cortina abierta"],
  ["cortinas", "cortinas"],
  ["cual", "cual"],
  ["cuando", "cuando"],
  ["cuanto", "cuanto"],
  ["cuarenta", "cuarenta"],
  ["cuarto", "cuarto"],
  ["cuatro", "cuatro"],
  ["cuerpo", "cuerpo"],
  ["cumpleanos", "cumpleaños"],
  ["cuna", "cuna"],
  ["dar", "dar"],
  ["dar pasos", "dar-pasos"],
  ["darse cuenta de", "darse cuenta de"],
  ["debajo", "debajo"],
  ["decir", "decir"],
  ["decirme", "decirme"],
  ["dentro", "dentro"],
  ["desaparecer", "desaparecer"],
  ["desaparecido", "desaparecido"],
  ["desayuno", "desayuno"],
  ["despues", "despues"],
  ["despues o siguiente", "después ó siguiente"],
  ["detalles caracteristicas perfiles", "detalles/ caracteristicas/ pérfiles"],
  ["dia", "dia"],
  ["dibujo", "dibujo"],
  ["dice", "dice"],
  ["diecinueve", "diecinueve"],
  ["dieciocho", "dieciocho"],
  ["dieciseis", "dieciseis"],
  ["diecisiete", "diecisiete"],
  ["diez", "diez"],
  ["dificil", "dificil", "difícil"],
  ["dijo", "dijo"],
  ["dinero", "dinero"],
  ["doce", "doce"],
  ["donde", "donde", "dónde"],
  ["dormir", "dormir"],
  ["dos", "dos"],
  ["dos se acercan", "dos se acercan"],
  ["dueno propiedad de alguien", "dueño(a)/ propiedad de alguien"],
  ["ejercicio", "ejercicio"],
  ["el", "el", "él"],
  ["el fue", "el fue"],
  ["el otro viene", "el otro viene"],
  ["ella", "ella"],
  ["ellos", "ellos"],
  ["empezar", "empezar"],
  ["encontrar", "encontrar"],
  ["encontrar acercar", "encontrar/ acercar"],
  ["encontrar o acercar", "encontrar o acercar"],
  ["encontrarse o acercarse", "encontrarse o acercarse"],
  ["enemigo", "enemigo"],
  ["enganar", "engañar"],
  ["engordar", "engordar"],
  ["entender", "entender"],
  ["entendiste?", "entendiste?"],
  ["entrar", "entrar"],
  ["entrar adentro", "entrar/ adentro"],
  ["esa ella", "esa/ ella"],
  ["esa mujer", "esa mujer"],
  ["escapar fugar", "escapar/ fugar"],
  ["esconder", "esconder"],
  ["esconderse", "esconderse"],
  ["escribir", "escribir"],
  ["escuchar", "escuchar"],
  ["ese hombre", "ese hombre"],
  ["ese? yo?", "ese? yo?"],
  ["esos", "esos", "ese", "ese"],
  ["espaguetis", "espaguetis"],
  ["espejo", "espejo"],
  ["esperar", "esperar"],
  ["espumadera", "espumadera"],
  ["estar bien", "estar-bien"],
  ["este", "este", "este(a)"],
  ["este esta", "este/ esta"],
  ["este esta ella", "este/esta/ella"],
  ["falta", "falta"],
  ["faltar", "faltar"],
  ["familia", "familia"],
  ["feliz", "feliz"],
  ["fin", "fin"],
  ["flaco", "flaco"],
  ["fortachon", "fortachon"],
  ["foto", "foto"],
  ["fregado", "fregado"],
  ["frio", "frio", "frío"],
  ["fue", "fue"],
  ["fundar"],
  ["futuro", "futuro"],
  ["g j o n", "g-j-o-n"],
  ["goma de mascar"],
  ["gordo"],
  ["gorro"],
  ["gracias"],
  ["grande"],
  ["guardar"],
  ["hablamos"],
  ["hablar"],
  ["hacer"],
  ["hacer preguntas", "hacer preguntas"],
  ["hambriento"],
  ["helicoptero"],
  ["hijo", "hijo", "hijo(a)"],
  ["historia"],
  ["hola"],
  ["hombre"],
  ["hoy"],
  ["hum?", "hum?", "hummm?", "hummm???", "hummm? sí"],
  ["idea"],
  ["igual" ],
  ["imposible"],
  ["infiel"],
  ["ir", "irse"],
  ["ix oculto", "ix-oculto"],
  ["j o n", "j-o-n"],
  ["j o n y s u e", "j-o-n-y-s-u-e"],
  ["j u a n", "j-u-a-n"],
  ["joven"],
  ["jovenes", "jovenes", "jovenes (chicas)", "jovenes chicos", "jovenes/chicos"],
  ["jugar"],
  ["juntos"],
  ["juntos en grupo", "juntos/ en grupo"],
  ["lancha",],
  ["leche"],
  ["leche dulce", "leche dulce"],
  ["lejos", "lejos"],
  ["lentes", "lentes"],
  ["lentes de sol", "lentes-de-sol", "lentes de sol oracion", "lentes-de-sol-oracion"],
  ["linterna", "linternas"],
  ["llamada", "llamar"],
  ["llegar", "llegar"],
  ["lo siento", "lo-siento"],
  ["lo tres", "lo tres", "los tres"],
  ["luego", "luego"],
  ["malo", "malo"],
  ["mandar", "mandar"],
  ["mapa", "mapa"],
  ["matrimonio", "matrimonio"],
  ["matrimonio boda", "matrimonio/ boda"],
  ["me dicen", "me dicen"],
  ["mejor", "mejor"],
  ["mi", "mi"],
  ["mochila", "mochila"],
  ["moneda", "moneda"],
  ["mucho", "mucho"],
  ["mucho dinero", "mucho-dinero"],
  ["mujer", "mujer", "mujeres", "mujeres"],
  ["musica", "música"],
  ["nacer", "nacer"],
  ["nada mas fin", "nada más/ fin"],
  ["navio", "navío"],
  ["ninguno", "ninguno"],
  ["nino", "niño"],
  ["no", "no", "no ", "no- no- no", "no-no", "no-no-no"],
  ["no importa", "no importa "],
  ["no sabe"],
  ["nombre"],
  ["nosotros", "nosotros"],
  ["nube", "nube", "nube clima", "nube/ clima"],
  ["o", "ó"],
  ["objetivo proposito", "objetivo/ proposito"],
  ["ok", "ok"],
  ["opaco", "opaco"],
  ["oscuro", "oscuro"],
  ["otro", "otro", "otro(a)", "otro uno", "otro uno"],
  ["oye", "oye"],
  ["p d", "p-d"],
  ["p e d r o", "p-e-d-r-o"],
  ["paciencia", "paciencia"],
  ["pais", "país"],
  ["panzon con rollos", "panzón / con rollos"],
  ["parada", "parada"],
  ["paragua", "paragua"],
  ["parecer", "parecer"],
  ["pelota", "pelota"],
  ["pensando o razonando", "pensar", "pensar razonar", "pensar/ razonar", "pensar razonar reflexionar", "pensar/ razonar/ reflexionar"],
  ["pequeno", "pequeño"],
  ["pequeno chico", "pequeño/ chico"],
  ["perder", "perder"],
  ["perfume", "perfume"],
  ["perseguir seguir", "perseguir/ seguir"],
  ["persona baja del bote", "persona baja del bote"],
  ["persona pequena", "persona-pequeña"],
  ["pescar", "pescar"],
  ["pez", "pez"],
  ["pila", "pila"],
  ["policia", "policia", "policía"],
  ["preguntar", "preguntar"],
  ["presente", "presente"],
  ["primero", "primero", "primero "],
  ["probar", "probar"],
  ["pronto en breve", "pronto/ en breve"],
  ["proposito objetivo", "próposito/ objetivo"],
  ["prueba", "prueba"],
  ["puerta", "puerta"],
  ["que", "que?", "qué"],
  ["que hace?", "qué hace?"],
  ["querer", "querer"],
  ["quien", "quien"],
  ["quinto", "quinto"],
  ["r a f a e l", "r-a-f-a-e-l"],
  ["razonando", "razonando"],
  ["rojo", "rojo"],
  ["ropa", "ropa"],
  ["rosado", "rosado"],
  ["s u e", "s-u-e"],
  ["saber", "saber"],
  ["salir vamos", "salir/ vamos"],
  ["saludan", "saludan"],
  ["se fue", "se fue"],
  ["se fue salio", "se fue/ salió", "se fue salir", "se fue/ salir"],
  ["se separa", "se separa"],
  ["segundo", "segundo"],
  ["sentir", "sentir"],
  ["sexto", "sexto"],
  ["si", "sí", "sí - sí - sí"],
  ["sol", "sol"],
  ["sordo", "sordo"],
  ["sorprendida se asombra", "sorprendida/ se asombra"],
  ["subir persona", "subir-persona"],
  ["sudar", "sudar"],
  ["suludos", "suludos"],
  ["telefono", "telefono"],
  ["tercero", "tercero"],
  ["termine", "terminé"],
  ["timida", "tímida"],
  ["timida con roche", "tímida/ con roche"],
  ["titulo", "titulo"],
  ["todo juntos", "todo juntos"],
  ["trampa", "trampa"],
  ["tranquila", "tranquila"],
  ["tres", "tres"],
  ["trotar correr", "trotar/ correr"],
  ["tu", "tú"],
  ["tu o el", "tú ó él"],

  ["un una", "un/ una"],
  ["uno", "uno", "un uno", "un? uno?", "un/ uno", "un/ uno(a)"],

  ["uruguay", "uruguay"],
  ["varios", "varios"],
  ["ver", "ver"],
  ["verde", "verde"],
  ["verguenza", "verguenza"],
  ["vio miro", "vió/ miró"],
  ["vio o miro", "vió ó miró"],
  ["viveres", "víveres"],
  ["ya", "ya"],
  ["ya se", "ya sé"],
  ["yo", "yo"],
  ["yogur", "yogur"],
]

BLEU_CORRECTOR_PROMPT = (
    "Recibes la transcripción aproximada, ruidosa y con posibles errores ortográficos,"
    " de un modelo que alinea embeddings de señas en español.\n"
    "Debes devolver únicamente la palabra o frase más probable en español neutro,"
    " sin explicaciones adicionales, sin comillas y en minúsculas.\n"
    "Transcripción aproximada: {noisy_text}\n"
    "Los tokens ruidosos del modelo aparecerán a continuación; úsalos solo como pista.\n"
    "Corrección:"
)

BLEU_SYSTEM_PROMPT = (
    "Eres un corrector ortográfico para modelos de señas."
    " Responde solo con la palabra o frase más probable en español neutro," 
    " sin signos adicionales ni explicaciones."
    " Si no entiendes la transcripción, responde con la transcripción tal cual. ejemplo:\n"
    " Transcripción aproximada: spag<pad<thi\n"
    " Corrección: spaghetti\n"
    " Transcripción aproximada: gomaĠdeĠmascar\n"
    " Corrección: goma de mascar\n"
    "Transcripción aproximada: aĠtierra\n"
    "Corrección: a tierra\n"
)

DEFAULT_BENCHMARK_DATASETS: Optional[List[str]] = ["dataset1", "dataset2", "dataset3", "dataset5"]


def build_corrector_prompt(noisy_text: str) -> str:
    cleaned = (noisy_text or "<pad>").strip()
    return BLEU_CORRECTOR_PROMPT.format(
        noisy_text=cleaned
    )

def bleu_collate_fn(batch):
    keypoints, mask_data, _, mask_embds, labels, dataset_tags = collate_fn(batch)
    return keypoints, mask_data, mask_embds, labels, dataset_tags

# --- NLTK BLEU (1/2/4) con smoothing, formato corpus_bleu ---
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def _tok(s: str) -> list[str]:
    # tokenización mínima y robusta para BLEU a nivel palabra
    return (s or "").strip().lower().split()

def _prepare_nltk_refs(refs: list[list[str]]) -> list[list[list[str]]]:
    """
    Convierte tu lista de variantes por muestra (List[str]) al formato NLTK:
    List[ sample -> List[reference -> List[tokens]] ]
    """
    return [[_tok(r) for r in ref_list] for ref_list in refs]

def _prepare_nltk_hyps(hyps: list[str]) -> list[list[str]]:
    """
    Convierte tus hipótesis a formato NLTK:
    List[ sample -> List[tokens] ]
    """
    return [_tok(h) for h in hyps]

def exec_nltk_bleu_all(
    refs_per_item: list[list[str]],
    hyps_per_item: list[str],
) -> dict[str, float|list[int]]:
    """
    Calcula BLEU-1, BLEU-2 y BLEU-4 (corpus-level) con smoothing (method1).
    refs_per_item: p.ej. [["a tierra","a tierra"], ["abecedario","abecedario.mp4"], ...]
    hyps_per_item: p.ej. ["a tierra", "abecedario", ...]
    """
    refs_nltk = _prepare_nltk_refs(refs_per_item)
    hyps_nltk = _prepare_nltk_hyps(hyps_per_item)

    smooth = SmoothingFunction().method1
    B1 = corpus_bleu(refs_nltk, hyps_nltk, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    B2 = corpus_bleu(refs_nltk, hyps_nltk, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    B4 = corpus_bleu(refs_nltk, hyps_nltk, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    return {"BLEU-1": B1, "BLEU-2": B2, "BLEU-4": B4}

def load_config():
    _,_, h5_file = setup_paths()
        
    training_cfg:dict = cfg.training
    model_cfg = cfg.model
    model_cfg["device"] = "cuda"
    return h5_file, training_cfg, model_cfg

def load_dataset(h5_file, key_points:int, allowed_datasets: Optional[list[str]] = None):
    allowed = allowed_datasets or DEFAULT_BENCHMARK_DATASETS
    keypoint_reader = KeypointDataset(
        h5Path=h5_file,
        return_label=True,
        return_dataset=True,
        n_keypoints=key_points,
        data_augmentation=False,
        allowed_datasets=allowed,
        labels_vocab_path= "../vocab_1235.json",
        max_length=4000,
    )
    train_dataset, _, _, _ = keypoint_reader.split_dataset(1)
    train_sampler = BatchSampler(train_dataset, 1)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=bleu_collate_fn,
        batch_sampler=train_sampler
    )
    return (
        train_dataloader,
        keypoint_reader.id_to_label,
        keypoint_reader.label_to_id,
        keypoint_reader.id_to_dataset,
    )

def load_model(model_parameters:dict, version:str, checkpoint:str, epoch:int):
    model_parameters.pop("device", None)  # remove device from model parameters
    print(model_parameters)
    model = build_model(**model_parameters)    

    model_location = f"../outputs/checkpoints/{version}/{checkpoint}/{epoch}/checkpoint.pth" 
    if not os.path.exists(model_location):
        raise FileNotFoundError(
            f"Model not found {model_location}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_location, map_location=device)

    model.load_state_dict(state_dict["model_state"])
    model.to(device)

    return model

def load_llm(model_id="unsloth/Llama-3.2-3B-Instruct", return_model: bool = False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # o "fp4"
        bnb_4bit_compute_dtype=torch.bfloat16,  # o torch.float16 si no tienes soporte bf16
    )

    device_map = "auto" if torch.cuda.is_available() else None
    llama_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    embedding_weight = cast(torch.Tensor, llama_model.get_input_embeddings().weight)
    embeddings = embedding_weight.detach().clone()
    returned_model = llama_model if return_model else None
    if not return_model:
        del llama_model
    return tokenizer, embeddings, returned_model

def get_idx_hyps(word):
    for i, cluster in enumerate(HYPS):
        if word in cluster:
            return i
    return -1

@torch.no_grad()
def embeddings_to_text_viterbi(
    embeddings: torch.Tensor,        # [L, D] (viene de tu modelo o del HDF5)
    all_embeddings: torch.Tensor,    # [V, D] (tabla del LLM)
    tokenizer,
    topk: int = 24,
    tau: float = 0.30,
    rep_penalty: float = 0.65,
    nospace_run_penalty: float = 0.10,
    start_space_bonus: float = 0.35,
):
    # Unificar device y dtype
    device = all_embeddings.device
    dtype  = all_embeddings.dtype

    # eps evita NaNs si llega un vector ~cero
    X = F.normalize(embeddings.to(device=device, dtype=dtype), p=2, dim=1, eps=1e-6)  # [L, D]
    E = F.normalize(all_embeddings.to(device=device, dtype=dtype), p=2, dim=1, eps=1e-6)  # [V, D]

    S = torch.matmul(X, E.T)  # [L, V]  <-- ya no crashea
    topv, topi = torch.topk(S, k=min(topk, S.size(1)), dim=1)

    special = set(tokenizer.all_special_ids)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        special.discard(bos_id)

    tok_str = tokenizer.convert_ids_to_tokens(torch.arange(E.size(0), device=device).tolist())

    # DP/Viterbi (igual que te pasé antes) ...
    # --- inicialización ---
    dp   = torch.full((topv.size(0), topv.size(1)), -1e9, device=device)
    prev = torch.full_like(dp, -1, dtype=torch.long)

    def is_space(tok: str) -> bool:
        return tok.startswith("▁")

    # t=0
    for j in range(topv.size(1)):
        vid = int(topi[0, j].item())
        if vid in special: 
            continue
        sc = float(topv[0, j].item())
        if sc < tau:
            continue
        if is_space(tok_str[vid]): 
            sc += start_space_bonus
        dp[0, j] = sc

    # transiciones
    for t in range(1, topv.size(0)):
        for j in range(topv.size(1)):
            vj = int(topi[t, j].item())
            if vj in special:
                continue
            base = float(topv[t, j].item())
            if base < tau:
                continue
            cur_space = is_space(tok_str[vj])

            best_val = -1e9
            best_k   = -1
            for k in range(topv.size(1)):
                prev_val = float(dp[t-1, k].item())
                if prev_val <= -1e8:
                    continue
                vi = int(topi[t-1, k].item())
                val = prev_val + base
                if vi == vj:
                    val -= rep_penalty
                prev_space = is_space(tok_str[vi])
                if not prev_space and not cur_space:
                    val -= nospace_run_penalty
                if val > best_val:
                    best_val = val
                    best_k   = k
            dp[t, j]   = best_val
            prev[t, j] = best_k

    # backtrack
    last_t = topv.size(0) - 1
    j = int(torch.argmax(dp[last_t]).item())
    if dp[last_t, j].item() <= -1e8:
        return ""

    ids = []
    for t in range(last_t, -1, -1):
        ids.append(int(topi[t, j].item()))
        j = int(prev[t, j].item())
        if t > 0 and j < 0:
            break
    ids.reverse()

    # limpieza
    ids2 = []
    for vid in ids:
        if vid in special or vid == bos_id:
            continue
        if not ids2 or ids2[-1] != vid:
            ids2.append(vid)

    pieces = tokenizer.convert_ids_to_tokens(ids2)
    text = ''.join(p.replace('▁', ' ') for p in pieces).strip()
    return text

def main(version:str, checkpoint:str, epoch:int, use_cached_results: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_file, training_cfg, model_cfg = load_config()
    dataset, id_to_label, label_to_id, id_to_dataset = load_dataset(
        h5_file, model_cfg.get("n_keypoints", 89)
    )
    if id_to_dataset:
        print(f"Datasets disponibles: {', '.join(id_to_dataset)}")
     
    model = load_model(model_cfg, version, checkpoint, epoch)
    model = model.to(device)
    model.eval()
    
    cache_path = "bleu_imitator_embeds.h5"
    dataset_size = len(dataset)
    results = []
    
    if use_cached_results and os.path.exists(cache_path):
        print("Found cached BLEU embeddings, loading...")
        with h5py.File(cache_path, "r") as h5f:
            samples_group = h5f.get("samples")
            stored_size = h5f.attrs.get("dataset_size")
            processed_size = h5f.attrs.get("processed_samples")
            if (
                isinstance(samples_group, h5py.Group)
                and len(samples_group) == dataset_size
                and stored_size == dataset_size
                and processed_size == dataset_size
            ):
                for key in sorted(samples_group.keys()):
                    sample_group = samples_group[key]
                    if not isinstance(sample_group, h5py.Group):
                        continue
                    label_value = sample_group.attrs.get("label", "")
                    if isinstance(label_value, bytes):
                        label_value = label_value.decode("utf-8")
                    dataset_value = sample_group.attrs.get("dataset", "")
                    if isinstance(dataset_value, bytes):
                        dataset_value = dataset_value.decode("utf-8")
                    dataset_value = dataset_value or None
                    hyps_ds = sample_group.get("hyps")
                    embed_ds = sample_group.get("embed_pred")
                    if not isinstance(hyps_ds, h5py.Dataset) or not isinstance(embed_ds, h5py.Dataset):
                        continue
                    hyps_data = hyps_ds[()]
                    hyps_list = hyps_data.tolist() if isinstance(hyps_data, np.ndarray) else list(hyps_data)
                    hyps_list = [item.decode("utf-8") if isinstance(item, bytes) else item for item in hyps_list]
                    results.append({
                        "label": label_value,
                        "dataset": dataset_value,
                        "embed_pred": embed_ds[()],
                        "hyps": hyps_list
                    })
                use_cached_results = True
                print("Loaded cached BLEU embeddings.")
    
    if not use_cached_results:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(cache_path, "w") as h5f:
            h5f.attrs["dataset_size"] = dataset_size
            samples_group = h5f.create_group("samples")

            for sample_idx, (keypoints, mask_data, mask_embds, label_id, dataset_tag) in enumerate(tqdm(dataset, desc="Processing samples")):
                label_text = id_to_label[label_id[0]]
                dataset_name = dataset_tag[0] if dataset_tag and len(dataset_tag) > 0 else None
                idx = get_idx_hyps(label_text)
                if idx == -1:
                    # print("no hay mapeado", label_text)
                    # añadir si no existe
                    HYPS.append([label_text])
                    idx = len(HYPS) - 1
                
                with torch.inference_mode():
                    data = keypoints.to(device=device, dtype=torch.float32, non_blocking=True)
                    mask_data = mask_data.to(device, non_blocking=True)

                    sign_embed, pool_embed = model(data, mask_data)
                    del pool_embed
                    
                    pred_embeds = sign_embed.to(device=device, dtype=torch.float32)
                    valid_tokens = (~mask_embds.to(pred_embeds.device, non_blocking=True)[0]).sum().item()
                    seq_len = min(pred_embeds.size(1), valid_tokens)
                    pred_embeds = pred_embeds[0, :seq_len, :].contiguous()

                    embed_array = pred_embeds.detach().cpu().numpy().astype("float32")
                    
                    res = {
                        "label": label_text,
                        "dataset": dataset_name,
                        "embed_pred": embed_array,
                        "hyps": HYPS[idx]
                    }
                    results.append(res)
                    
                    sample_group = samples_group.create_group(f"{sample_idx:06d}")
                    sample_group.attrs["label"] = label_text
                    sample_group.attrs["dataset"] = dataset_name or ""
                    sample_group.create_dataset("embed_pred", data=embed_array, compression="gzip")

                    hyps_array = np.array(res["hyps"], dtype=object)
                    sample_group.create_dataset("hyps", data=hyps_array, dtype=string_dtype)
                    
                    del data, mask_data, sign_embed, pred_embeds
                    gc.collect()
                    torch.cuda.empty_cache()

            h5f.attrs["processed_samples"] = len(samples_group)
    
    # Liberar memoria de Imitator antes de cargar el LLM corrector
    del model
    torch.cuda.empty_cache()

    tokenizer_llm, all_embeddings, llama_model = load_llm(return_model=True)
    if llama_model is None:
        raise RuntimeError("LLM model not returned; ensure return_model=True in load_llm call")
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    llm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corrector = MultimodalSignLM(llama_model, tokenizer_llm, llm_device)

    dataset_refs = defaultdict(list)
    dataset_preds = defaultdict(list)

    for res in results:
        embed_pred = torch.from_numpy(res["embed_pred"]).to(
            device=all_embeddings.device,
            dtype=all_embeddings.dtype,
        )

        noisy_text = embeddings_to_text_viterbi(embed_pred, all_embeddings, tokenizer_llm)
        res["noisy_text"] = noisy_text

        prompt = build_corrector_prompt(noisy_text)
        corrected_text = corrector.generate_corrected_text(
            None,
            prompt,
            max_new_tokens=130,
            fallback_text=noisy_text,
            system_prompt=BLEU_SYSTEM_PROMPT,
        )
        corrected_text = corrected_text.replace("\n", " ").strip()
        corrected_text = corrected_text.split("<pad>")[0].strip()
        corrected_text = corrected_text.lower()
        res["pred_text"] = corrected_text

        dataset_name = res.get("dataset") or "unknown"
        dataset_refs[dataset_name].append(res["hyps"])
        dataset_preds[dataset_name].append(corrected_text)

    hyps_list_bench = [res["hyps"] for res in results]
    pred_list_bench = [res["pred_text"] for res in results]

    print(hyps_list_bench[:5], pred_list_bench[:5])
    score_bleu = exec_nltk_bleu_all(hyps_list_bench, pred_list_bench)
    print(f"BLEU score (global): {score_bleu}")

    for dataset_name in sorted(dataset_preds.keys()):
        nltk_ds = exec_nltk_bleu_all(dataset_refs[dataset_name], dataset_preds[dataset_name])
        print(f"NLTK BLEU {dataset_name} [{len(dataset_preds[dataset_name])}]: "
            f"B1={nltk_ds['BLEU-1']:.4f}  "
            f"B2={nltk_ds['BLEU-2']:.4f}  "
            f"B4={nltk_ds['BLEU-4']:.4f}")
    
    with open("results.txt", "w") as f:
        for res in results:
            dataset_name = res.get("dataset") or "unknown"
            pred = res["pred_text"]
            hyps = res["hyps"]
            clean_pred = pred.split("<pad>")[0].strip()
            noisy_text = res.get("noisy_text", "")
            f.write(
                f"DATASET: {dataset_name}\tHYPS: {hyps}\tNOISY: {noisy_text}\tPRED: {pred}\tCLEAN PRED: {clean_pred}\n"
            )
    del llama_model
    torch.cuda.empty_cache()
    return score_bleu