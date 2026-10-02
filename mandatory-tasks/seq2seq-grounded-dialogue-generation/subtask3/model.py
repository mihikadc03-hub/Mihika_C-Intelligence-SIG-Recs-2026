import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class DualSourceDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention_dialogue, attention_document):
        super().__init__()
        self.output_dim = output_dim
        self.attention_dialogue = attention_dialogue
        self.attention_document = attention_document
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Input to GRU is embedded token + context from dialogue + context from document
        self.rnn = nn.GRU((enc_hid_dim * 2 * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2 * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, dial_enc_outputs, doc_enc_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        # Dialogue attention
        a_dial = self.attention_dialogue(hidden, dial_enc_outputs)
        a_dial = a_dial.unsqueeze(1)
        dial_enc_outputs_p = dial_enc_outputs.permute(1, 0, 2)
        weighted_dial = torch.bmm(a_dial, dial_enc_outputs_p)
        weighted_dial = weighted_dial.permute(1, 0, 2)
        
        # Document attention
        a_doc = self.attention_document(hidden, doc_enc_outputs)
        a_doc = a_doc.unsqueeze(1)
        doc_enc_outputs_p = doc_enc_outputs.permute(1, 0, 2)
        weighted_doc = torch.bmm(a_doc, doc_enc_outputs_p)
        weighted_doc = weighted_doc.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted_dial, weighted_doc), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_dial = weighted_dial.squeeze(0)
        weighted_doc = weighted_doc.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted_dial, weighted_doc, embedded), dim=1))
        
        return prediction, hidden.squeeze(0)

class GroundedSeq2Seq(nn.Module):
    def __init__(self, dialogue_encoder, document_encoder, decoder, device):
        super().__init__()
        self.dialogue_encoder = dialogue_encoder
        self.document_encoder = document_encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, dial_src, doc_src, trg, teacher_forcing_ratio=0.5):
        batch_size = dial_src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        dial_enc_outputs, dial_hidden = self.dialogue_encoder(dial_src)
        doc_enc_outputs, doc_hidden = self.document_encoder(doc_src)
        
        # We can combine hidden states or just use the dialogue hidden state as the initial decoder state
        hidden = dial_hidden 
        
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, dial_enc_outputs, doc_enc_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs
