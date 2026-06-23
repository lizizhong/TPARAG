# TPARAG
This repository reuses the code from [Atlas](https://github.com/facebookresearch/atlas) but replaces the language model structure in Atlas from encoder-to-decoder to decoder-only, serving as the code base for the anaysis part of paper [Token-Level Precise Attack on RAG: Searching for the Best Alternatives to Mislead Generation](https://aclanthology.org/2024.naacl-short.65.pdf)

![TPARAG](./img/TPARAG_main.pdf)
<center>The framework of our proposed TPARAG. TPARAG first generates parent malicious passages through the generation attack stage (left). These passages are then recombined and refined during the optimization attack stage (right), producing optimized malicious passages that effectively mislead RAG's answer.</center>

The code is mainly divided into two parts: *generation attack* and *optimization attack*. Please run `optimization_attack.py` directly to execute TPARAG.
Usage:
    python optimization_attack.py \
        --iteraction_number 5 \
        --instance_filename original_retrieved_passage_file \
        --write_filename result_file \
        --open_source True \
        --setting_type black-box \
        --candidates_number 20 \
        --pos_substution_rate 0.2 

The previous section before running TPARAG (e.g., how to retrieve the original passage set) and the packages requirements will be added shortly.
