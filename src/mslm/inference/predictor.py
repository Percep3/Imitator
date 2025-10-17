import torch
import torch.nn.functional as F
from typing import Optional, List, Dict
from transformers import LogitsProcessor

printed = False

class BanEOTFirstStep(LogitsProcessor):
    def __init__(self, eot_id: int):
        self.eot_id = eot_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Si estamos generando el primer token nuevo, prohíbe EOT
        step = input_ids.shape[1]  # longitud actual
        if step == input_ids.shape[1]:  # (HF ya está en paso nuevo)
            scores[:, self.eot_id] = -float("inf")
        return scores

def _utf_fix(s: str) -> str:
    # 2) Limpieza rápida de mojibake y ruido
    try:
        import ftfy
        s = ftfy.fix_text(s)
    except Exception:
        try:
            s = s.encode("latin1").decode("utf-8")
        except Exception:
            pass
    s = s.replace("<pad>", " ").strip().lower()
    import unicodedata, re
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s

class MultimodalSignLM:
    def __init__(self, base_model, tokenizer, device):
        """
        Initialize the MultimodalSignLM class.
        Params
        :model: LLama 3 model.
        :tokenizer: The tokenizer for the model.
        :device: The device to run the model on (e.g., 'cuda' or 'cpu').
        """

        self.model = base_model
        self.tokenizer = tokenizer
        self.device = device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Get the embeddings of all tokens in the vocabulary
        self.all_embeddings = base_model.get_input_embeddings().weight.data.to(self.device)

    def process_inputs(self, keypoints_embeddings, text_input:str):
        # Preprocess the inputs
        keypoints_embeddings = keypoints_embeddings.to(self.device)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        
        sign_embed_tokens = torch.tensor(
            [self._find_closest_token(emb, self.all_embeddings) for emb in keypoints_embeddings[0]]  # embeddings[0] porque es un batch de tamaño 1
        ).unsqueeze(0).to(self.device)

        # Add EOS token at the end of the keypoints
        eos_token_id = self.tokenizer.eos_token_id
        sign_embed_tokens = torch.cat([sign_embed_tokens, torch.tensor([[eos_token_id]]).to(self.device)], dim=1)

        # Concatenate the token embeddings with the text input
        inputs['input_ids'] = torch.cat([inputs['input_ids'], sign_embed_tokens], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(sign_embed_tokens)], dim=1)
        
        return inputs
    
    def generate(self, keypoints_embeddings, text_input: str, max_new_tokens: int = 64):
        self.model.eval()

        chat_format = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 07 Apr 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        # Process the inputs
        inputs = self.process_inputs(keypoints_embeddings, chat_format + text_input)

        # Generate the output
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the output
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=False)
        return decoded_output

    def _extract_assistant_response(self, generated_text: str) -> str:
        """Extrae la respuesta del asistente del formato de chat de Llama."""
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in generated_text:
            generated_text = generated_text.split(marker, 1)[-1]
        if "<|eot_id|>" in generated_text:
            generated_text = generated_text.split("<|eot_id|>", 1)[0]
        return generated_text.strip()

    def generate_corrected_text(
        self,
        keypoints_embeddings: Optional[torch.Tensor],
        prompt: str,
        max_new_tokens: int = 64,
        fallback_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Genera una corrección textual empleando el LLM en formato chat."""
        _ = keypoints_embeddings  # se conserva la firma para compatibilidad
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        raw_output = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        global printed
        if not printed:
            print(f"Raw output: {raw_output} END RAW")
            printed = True
        
        response = self._extract_assistant_response(raw_output)
        response = response.strip()
        if not response and fallback_text is not None:
            return fallback_text
        return response
    
    def _find_closest_token(self, embedding, all_embeddings):
        embedding = embedding.to(self.device)
        if embedding.dim() > 1:
            embedding = embedding.squeeze()

        # Calcular similitud del coseno
        similarities = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings, dim=1)

        # Encontrar el índice del token más similar
        closest_token_id = torch.argmax(similarities).item()
        return closest_token_id

    def embeddings_to_text(self, embeddings: torch.Tensor) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()

        embeddings = embeddings.to(device)

        embedding_matrix = self.all_embeddings.to(device)  # [V, D]

        embedding_matrix_norm = F.normalize(embedding_matrix, p=2, dim=1)  # [V, D]
        print(embedding_matrix_norm.shape)

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [T, D]

        similarities = torch.matmul(embeddings_norm, embedding_matrix_norm.T)  # [T, V]

        token_ids = torch.argmax(similarities, dim=1).tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)