
def adjust_config(config, tokenizer, args):
    # import pdb; pdb.set_trace()
    config.pad_token_id = tokenizer.pad_token_id
    config.decoder.pad_token_id = tokenizer.pad_token_id
    config.decoder.bos_token_id = tokenizer.sep_token_id
    config.decoder.eos_token_id = tokenizer.sep_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.enable_phrase_retrieval = args['enable_phrase_retrieval']
    config.src_vocab_size = tokenizer.vocab_size
    config.tgt_vocab_size = tokenizer.vocab_size
    config.decoder.vocab_size = tokenizer.vocab_size
    if 'enable_retriever_encoder' in args:
        config.enable_retriever_encoder = args['enable_retriever_encoder']
    if 'd_model' in args:
        config.d_model = args['d_model']
    if 'tie_word_embeddings' in args:
        config.tie_word_embeddings = args['tie_word_embeddings']
    if 'phrase_hidden_size' in args:
        config.phrase_hidden_size = args['phrase_hidden_size']
    else:
        config.phrase_hidden_size = 128
    if 'encoder_config_dir' in args:
        config.encoder_config_dir = "{}/{}".format(args['root_dir'], args['encoder_config_dir']) 
    if 'phrase_encoder_type' in args:
        config.phrase_encoder_type = args['phrase_encoder_type']
    if 'enable_phrase_pe' in args:
        config.enable_phrase_pe = args['enable_phrase_pe']
    else:
        config.enable_phrase_pe = False
    if 'enable_phrase_reencoding' in args:
        config.enable_phrase_reencoding = args['enable_phrase_reencoding']
    else:
        config.enable_phrase_reencoding = False

    if 'reencoding_with_cls_token' in args:
        config.reencoding_with_cls_token = args['reencoding_with_cls_token']
    else:
        config.reencoding_with_cls_token = False
    
    # NOTE: `phrase_modeling_method` should be in [`attn`, `concat`]
    config.phrase_modeling_method = args.get('phrase_modeling_method', 'attn')

    return config

def override(args, override_args=None):
    if override_args is not None:
        if isinstance(override_args, str):
            override_args = eval(override_args)
    elif 'override_args' in args and args['override_args']:
        override_args = eval(args['override_args'])
    else:
        return args
    assert isinstance(override_args, dict)
    for k, v in override_args.items():
        if k in args:
            if v != args[k]:
                print('[!] override `{}`: {} -> {}'.format(k, args[k], v))
                args[k] = v
            else:
                print('[!] already exist `{}`: {}'.format(k, v))
        else:
            print('[!] add new arg `{}`: {}'.format(k, v))
            args[k] = v
    return args