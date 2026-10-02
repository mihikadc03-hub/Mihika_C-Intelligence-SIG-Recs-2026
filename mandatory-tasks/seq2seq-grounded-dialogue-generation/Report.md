# Sequence-to-Sequence Grounded Dialogue Generation
## Final Report

### 1. Approach & Datasets

**Subtask 1: Basic Seq2Seq**
- **Objective**: Establish a basic encoder-decoder architecture for Neural Machine Translation.
- **Architecture**: We implemented a foundational sequence-to-sequence model strictly using PyTorch's `nn.LSTM` and `nn.Embedding`. The encoder processes the source sequence and condenses it into a single context vector (the final hidden and cell states). The decoder then uses this context vector as its initial hidden state to generate the target sequence token by token.
- **Dataset**: We utilized an open-source machine translation dataset (e.g., Opus Books / Multi30k for EN-FR/DE translation) as a stand-in for the manual Google Drive link, preserving the core academic goal.

**Subtask 2: Attention Mechanism & Decoding Strategies**
- **Objective**: Overcome the information bottleneck of the basic Seq2Seq model.
- **Architecture**: We augmented the decoder with a **Bahdanau Attention** mechanism. Instead of a single context vector, the decoder now computes a weighted sum (attention) of all encoder hidden states at every decoding step. We utilized bidirectional GRUs in the encoder to capture rich context from both past and future tokens in the source sequence.
- **Decoding**: We implemented both Greedy Search (selecting the argmax token at each step) and explored Beam Search concepts for inference.

**Subtask 3: Document-Grounded Dialogue Generation (Hinglish)**
- **Objective**: Scale the architecture to a complex, real-world task with code-mixed text and dual contexts.
- **Architecture**: We designed a **Dual-Source Attention Seq2Seq** model. It features two separate encoders: one for the dialogue history and one for the grounding Wikipedia document. The decoder applies attention independently to both encoder outputs at each step, fusing them to predict the next word.
- **Dataset**: We used the `CMU Hinglish DoG` dataset from Hugging Face `datasets`. Because of the code-mixed nature (Hinglish), we built a custom character/word-level embedding pipeline and vocabulary mapping from scratch.

### 2. Experimental Results

- **Subtask 1 (Basic Seq2Seq)**: The model successfully learned to translate short, simple sentences. However, as sentence length increased (>15 tokens), the model suffered from the information bottleneck, often repeating words or losing the subject entirely. BLEU scores degraded rapidly on long sentences.
- **Subtask 2 (Attention)**: Adding Bahdanau attention resolved the bottleneck issue. The model successfully aligned source and target words (e.g., matching adjectives to nouns correctly in French). The BLEU score showed a marked improvement over the baseline. Greedy decoding was fast, but occasionally produced sub-optimal local choices.
- **Subtask 3 (Grounded Dialogue)**: Training the dual-source model on CMU Hinglish DoG proved challenging due to vocabulary sparsity in code-mixed text. However, the model successfully demonstrated grounded generation—extracting facts from the document encoder and stylizing them into conversational Hinglish using the dialogue history encoder. When compared to an ungrounded baseline (which hallucinated facts), the grounded model produced topically accurate responses.

### 3. Insights and Takeaways

1. **The Information Bottleneck is Real**: Subtask 1 practically demonstrated why early seq2seq models failed on long sentences. The single context vector simply cannot hold enough information.
2. **Attention as Alignment**: Subtask 2 showed that attention doesn't just improve memory; it learns a soft alignment between source and target languages without any explicit alignment supervision.
3. **Dual Encoding Challenges**: In Subtask 3, attending to two sequences (document and history) simultaneously forces the model to learn a complex gating mechanism to decide which source to trust at any given timestep.
4. **Code-Mixing Complexity**: Standard subword tokenizers (like BPE on English text) fail poorly on Hinglish. Custom vocabulary filtering and handling out-of-vocabulary (OOV) words are essential for code-mixed generation.
