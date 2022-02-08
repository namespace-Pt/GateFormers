    # ids = np.memmap(
    #     os.path.join(cache_dir, "ids.mmp"),
    #     shape=(news_num, ), mode='r+', dtype=np.int32)
    # title_token_ids = np.memmap(
    #     os.path.join(cache_dir, "title_token_ids.mmp"),
    #     shape=(news_num, max_title_length), mode='r+', dtype=np.int32)
    # title_attn_masks = np.memmap(
    #     os.path.join(cache_dir, "title_attn_masks.mmp"),
    #     shape=(news_num, max_title_length), mode='r+', dtype=np.bool8)
    # title_dedup_masks = np.memmap(
    #     os.path.join(cache_dir, "title_dedup_masks.mmp"),
    #     shape=(news_num, max_title_length), mode='r+', dtype=np.bool8)
    # title_punc_masks = np.memmap(
    #     os.path.join(cache_dir, "title_punc_masks.mmp"),
    #     shape=(news_num, max_title_length), mode='r+', dtype=np.bool8)

    # abs_token_ids = np.memmap(
    #     os.path.join(cache_dir, "abs_token_ids.mmp"),
    #     shape=(news_num, max_abs_length), mode='r+', dtype=np.int32)
    # abs_attn_masks = np.memmap(
    #     os.path.join(cache_dir, "abs_attn_masks.mmp"),
    #     shape=(news_num, max_abs_length), mode='r+', dtype=np.bool8)
    # abs_dedup_masks = np.memmap(
    #     os.path.join(cache_dir, "abs_dedup_masks.mmp"),
    #     shape=(news_num, max_abs_length), mode='r+', dtype=np.bool8)
    # abs_punc_masks = np.memmap(
    #     os.path.join(cache_dir, "abs_punc_masks.mmp"),
    #     shape=(news_num, max_abs_length), mode='r+', dtype=np.bool8)