from .AMID_IF_util import *
import mathAMID_IF

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [Batch, 1], [0, 1]
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :] * 1000.0 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AMID_IF(nn.Module):
    def __init__(self, 
                num_letters=21, 
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                vocab=22,  
                k_neighbors=32,
                augment_eps=0.1,
                dropout=0.1):
        super().__init__()
        
        self.net_a = SingleStream(
            num_letters=num_letters,
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            vocab=vocab,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout
        )
        
        self.net_b = SingleStream_P(
            num_letters=num_letters,
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            vocab=vocab,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout
        )
        
        self.cross_attn_layers = nn.ModuleList([
            AlignedCrossAttention(hidden_dim=hidden_dim)
            for _ in range(num_encoder_layers)
        ])

        self.cross_dec_attn_layers = nn.ModuleList([
            AlignedCrossAttention(hidden_dim=hidden_dim)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p, align_pairs, tms):
        B, N = X.size(0), X.size(1)

        E_a, E_idx_a = self.net_a.features(X, mask, residue_idx, chain_encoding_all) 
        E_b, E_idx_b, h_V_b, h_E_b = [], [], [], []
        num_pairs = len(X_p)
        
        for k in range(num_pairs):
            eb, ei = self.net_b.features(X_p[k], mask_p[k], residue_idx_p[k], chain_encoding_all_p[k])
            E_b.append(eb)
            E_idx_b.append(ei)
        
        h_V_a, h_E_a = self.net_a.init_states(E_a, S)
        for k in range(num_pairs):
            hv, he = self.net_b.init_states_pair(E_b[k], S_p[k], self.net_a.W_s)
            h_V_b.append(hv)
            h_E_b.append(he)
        
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx_a).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        1
        aln_idx = []
        tm_score = []
        for k in range(num_pairs):
            tm_score.append([b[k] for b in tms]) 
            aln_idx.append([b[k] for b in align_pairs])

        tm_score_torch = torch.tensor(tm_score, device=X.device, dtype=torch.float32)
        tm_sum = tm_score_torch.sum(dim=0, keepdim=True)      
        tm_ratio = tm_score_torch / (tm_sum + 1e-8)           
            
        for k in range(num_pairs):
            for i in range(B):
                a_indices = [p[0] for p in aln_idx[k][i]]
                b_indices = [p[1] for p in aln_idx[k][i]]
                h_V_a[i,a_indices,:] += h_V_b[k][i,b_indices,:] * tm_score[k][i] * tm_ratio[k][i]

        for (enc_a, enc_b), cross_attn in zip(zip(self.net_a.encoder_layers, self.net_b.encoder_layers), self.cross_attn_layers):
            h_V_a, h_E_a = enc_a(h_V_a, h_E_a, E_idx_a, mask, mask_attend)
            for k in range(num_pairs): 
                h_V_b[k], h_E_b[k] = enc_b(h_V_b[k], h_E_b[k], E_idx_b[k], mask_p[k])
                h_V_a, _ = cross_attn(h_V_a, h_V_b[k], mask, aln_idx[k], tm_score[k])

        t = torch.rand((B, 1), device=X.device)
        mask_prob = 1.0 - t 
        
        rand_tensor = torch.rand((B, N), device=X.device)
        masked_indices = rand_tensor < mask_prob 
        
        X_t = S.clone()
        MASK_TOKEN = 21 
        X_t[masked_indices] = MASK_TOKEN
        X_t = X_t * mask.long()

        logits = self.net_a.forward_decoder_dfm(
            h_V_enc=h_V_a, 
            h_E=h_E_a, 
            E_idx=E_idx_a, 
            mask=mask, 
            X_t=X_t, 
            t=t,
            cross_attn_layers=self.cross_dec_attn_layers,
            h_V_b=h_V_b[0] if num_pairs > 0 else None,
            align_pairs=aln_idx[0] if num_pairs > 0 else None,
            tm_scores=tm_score[0] if num_pairs > 0 else None
        )
        
        return F.log_softmax(logits, dim=-1), masked_indices

    @torch.no_grad()
    def sample(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p, align_pairs, tms, steps=10, temperature=1.0):

        B, N = X.size(0), X.size(1)
        E_a, E_idx_a = self.net_a.features(X, mask, residue_idx, chain_encoding_all)
        E_b, E_idx_b, h_V_b, h_E_b = [], [], [], []
        num_pairs = len(X_p)
        # if num_pairs > 0:
        #     num_pairs = 1

        for k in range(num_pairs):
            eb, ei = self.net_b.features(X_p[k], mask_p[k], residue_idx_p[k], chain_encoding_all_p[k])
            E_b.append(eb)
            E_idx_b.append(ei)

        h_V_a, h_E_a = self.net_a.init_states(E_a)
        for k in range(num_pairs):
            hv, he = self.net_b.init_states_pair(E_b[k], S_p[k], self.net_a.W_s)
            h_V_b.append(hv)
            h_E_b.append(he)
        
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx_a).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        aln_idx = []
        tm_score = []
        for k in range(num_pairs):
            tm_score.append([b[k] for b in tms])
            aln_idx.append([b[k] for b in align_pairs])

        # The first feature fusion: Feature hard injection
        # Before the encoder starts, the representation of the template corresponding to the residue is embedded and added to the target corresponding to the TM-score weight.
        tm_score_torch = torch.tensor(tm_score, device=X.device, dtype=torch.float32)
        tm_sum = tm_score_torch.sum(dim=0, keepdim=True)      # [1, B]
        tm_ratio = tm_score_torch / (tm_sum + 1e-8)           # [num_pairs, B]

        for k in range(num_pairs):
            for i in range(B):
                a_indices = [p[0] for p in aln_idx[k][i]]
                b_indices = [p[1] for p in aln_idx[k][i]]
                h_V_a[i,a_indices,:] += h_V_b[k][i,b_indices,:] * tm_score[k][i] * tm_ratio[k][i]

        for (enc_a, enc_b), cross_attn in zip(zip(self.net_a.encoder_layers, self.net_b.encoder_layers), self.cross_attn_layers):
            h_V_a, h_E_a = enc_a(h_V_a, h_E_a, E_idx_a, mask, mask_attend)
            for k in range(num_pairs):
                h_V_b[k], h_E_b[k] = enc_b(h_V_b[k], h_E_b[k], E_idx_b[k], mask_p[k])
                h_V_a, _ = cross_attn(h_V_a, h_V_b[k], mask, aln_idx[k], tm_score[k])

        MASK_TOKEN = 21
        
        X_t = torch.full((B, N), MASK_TOKEN, device=X.device, dtype=torch.long)
        X_t = X_t * mask.long() 
        

        X_t[chain_M == 0] = S[chain_M == 0] 
        
        unmasked_tracker = (chain_M == 0) 
        
        for k in range(1, steps + 1):
            t_val = (k - 1) / steps
            t = torch.full((B, 1), t_val, device=X.device)
            

            mask_ratio = math.cos((k / steps) * (math.pi / 2)) 
            

            logits = self.net_a.forward_decoder_dfm(
                h_V_enc=h_V_a, h_E=h_E_a, E_idx=E_idx_a, mask=mask, 
                X_t=X_t, t=t,
                cross_attn_layers=self.cross_dec_attn_layers,
                h_V_b=h_V_b[0] if num_pairs>0 else None,
                align_pairs=aln_idx[0] if num_pairs>0 else None,
                tm_scores=tm_score[0] if num_pairs>0 else None
            )
            
            # probs = F.softmax(logits / temperature, dim=-1)
            # pred_classes = torch.multinomial(probs.view(-1, 20), 1).view(B, N)
            

            # confidence = torch.gather(probs, 2, pred_classes.unsqueeze(-1)).squeeze(-1)

            logits_aa = logits[:, :, :20] 
            probs_aa = F.softmax(logits_aa / temperature, dim=-1)
            

            pred_classes = torch.multinomial(probs_aa.view(-1, 20), 1).view(B, N)
            

            probs_full = F.softmax(logits / temperature, dim=-1)
            confidence = torch.gather(probs_full, 2, pred_classes.unsqueeze(-1)).squeeze(-1)
            

            confidence[unmasked_tracker] = float('inf')
            confidence[mask == 0] = float('inf') 
            

            num_to_mask = int(mask_ratio * torch.sum(chain_M[0])) 
            
            if num_to_mask > 0:

                _, lowest_conf_idx = torch.topk(confidence, num_to_mask, dim=-1, largest=False)
                
                X_t_new = pred_classes.clone()
                X_t_new.scatter_(1, lowest_conf_idx, MASK_TOKEN)
                X_t_new[unmasked_tracker] = X_t[unmasked_tracker] 
                X_t = X_t_new
            else:
                X_t[~unmasked_tracker] = pred_classes[~unmasked_tracker]

            unmasked_tracker = (X_t != MASK_TOKEN)

        return X_t

class SingleStream(nn.Module):
    def __init__(self, 
                num_letters=21, 
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                vocab=22, 
                k_neighbors=32,
                augment_eps=0.1,
                dropout=0.1):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, CA_only = False
        )
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.time_emb = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout) for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout) for _ in range(num_decoder_layers)
        ])
        
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_states(self, E, S=None):
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        return h_V, h_E
    
    def forward_decoder_dfm(self, h_V_enc, h_E, E_idx, mask, X_t, t, cross_attn_layers=None, h_V_b=None, align_pairs=None, tm_scores=None):

        h_S = self.W_s(X_t)
        t_embed = self.time_mlp(self.time_emb(t)).unsqueeze(1)
        h_V = h_V_enc + h_S + t_embed


        for i, layer in enumerate(self.decoder_layers):
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx) 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = layer(h_V, h_ESV, mask)
            
            if cross_attn_layers and h_V_b is not None:
                cross_attn = cross_attn_layers[i]
                h_V, _ = cross_attn(h_V, h_V_b, mask, align_pairs, tm_scores)
        
        return self.W_out(h_V)

class SingleStream_P(nn.Module):
    def __init__(self, 
                num_letters=21,
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                vocab=22,
                k_neighbors=32,
                augment_eps=0.1,
                dropout=0.1):
        super().__init__()
        

        self.hidden_dim = hidden_dim
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=4, augment_eps=augment_eps, CA_only = True
        )
        self.W_e = nn.Linear(edge_features, hidden_dim)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout) for _ in range(num_encoder_layers)
        ])
 
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_states_pair(self, E, S, w_s):
        h_V = initialize_h_V(self, E, S, w_s, onehot_dim=21, init_ratio=1.0, error_init_ratio=0.0) 
        h_E = self.W_e(E)
        return h_V, h_E

class AlignedCrossAttention(nn.Module):
    """A layer that performs attention only at predefined aligned positions"""
    def __init__(self, hidden_dim, attn_dim=256, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_a, h_b, mask, align_pairs_list, tm_scores=None):
        batch_size = h_a.size(0)
        updates = torch.zeros_like(h_a)  

        for i in range(batch_size):
            align_pairs = align_pairs_list[i]  
            a_indices = [p[0] for p in align_pairs]
            b_indices = [p[1] for p in align_pairs]
            
            Q = self.query(h_a[i, a_indices, :])  
            K = self.key(h_b[i, b_indices, :])    
            V = self.value(h_b[i, b_indices, :])  
            
            attn = torch.einsum('ph,ph->p', Q, K) * self.scale
            attn = F.softmax(attn, dim=0)

            updates[i, a_indices, :] = V * attn.unsqueeze(-1) * tm_scores[i]

        return h_a + self.dropout(updates), h_b