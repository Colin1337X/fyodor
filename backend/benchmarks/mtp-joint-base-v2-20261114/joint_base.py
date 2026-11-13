"""Isolated greedy MTP experiment: verify the base and proposals together."""
def transform(source):
    start = source.index('int nya_llm_generate_draft(')
    before, body = source[:start], source[start:]
    def replace(old, new):
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    replace('uint32_t *prompt = NULL, proposals[32], previous;',
            'uint32_t *prompt = NULL, proposals[32], previous, joint_tokens[33];\n    int joint_base = 0;\n    size_t joint_original_window = 0;')
    replace('''        if(nya_llm_cpu_product(window,vocabulary,&count) ||''', '''        joint_base = draft->is_assistant && r->temperature == 0.0f &&
            r->speculative_tokens == 4 && window >= 2;
        joint_original_window = window;
        if(joint_base) window = window > 4 ? 3 : window - 1;
        if(nya_llm_cpu_product(window + (joint_base ? 1u : 0u),vocabulary,&count) ||''')
    replace('''            if (nya_llm_forward(c, &target_state, base_token, position, 1) != 0) goto invalid;
            ++out->target_steps;''', '''            if(joint_base) joint_tokens[0] = base_token;
            else {
                if (nya_llm_forward(c, &target_state, base_token, position, 1) != 0) goto invalid;
                ++out->target_steps;
            }''')
    replace('if(early_enabled && proposed==0 && r->temperature==0.0f)',
            'if(!joint_base && early_enabled && proposed==0 && r->temperature==0.0f)')
    replace('''            memcpy(target_state.device_tokens+position,proposals,proposed*sizeof(*proposals));
            if(nya_compute_plan_verify(target_state.device_plan,proposals,proposed,position,
                p_bank,window*vocabulary)) goto invalid;
            /* Count every computed target position, including discarded work. */
            out->target_steps+=proposed;''', '''            size_t verify_count = proposed, verify_position = position;
            const uint32_t *verify_tokens = proposals;
            if(joint_base) {
                memcpy(joint_tokens + 1, proposals, proposed * sizeof(*proposals));
                ++verify_count;
                --verify_position;
                verify_tokens = joint_tokens;
            }
            memcpy(target_state.device_tokens+verify_position,verify_tokens,verify_count*sizeof(*verify_tokens));
            if(nya_compute_plan_verify(target_state.device_plan,verify_tokens,verify_count,verify_position,
                p_bank,(window + (joint_base ? 1u : 0u))*vocabulary)) goto invalid;
            /* Include the jointly verified base and discarded suffix positions. */
            out->target_steps+=verify_count;
            if(joint_base) {
                memcpy(target_state.logits, p_bank, vocabulary * sizeof(float));
                memmove(p_bank, p_bank + vocabulary, proposed * vocabulary * sizeof(float));
            }''')
    replace('''    previous = prompt[prompt_count - 1];''', '''    if(joint_base && trace && !strcmp(trace,"1"))
        fprintf(stderr,"MTP joint base: enabled; proposal window=%zu\\n",window);
    previous = prompt[prompt_count - 1];''')
    replace('    position = prompt_count; out->target_steps = prompt_count;', """    /* Prefill may invalidate the original resident plan and fall back. */
    if(joint_base && (!target_state.device_plan || target_state.reference_only ||
       !nya_compute_plan_can_verify(target_state.device_plan) ||
       !nya_compute_plan_can_rewind(target_state.device_plan) ||
       !nya_compute_plan_can_borrow(target_state.device_plan))) {
        joint_base = 0;
        window = joint_original_window;
    }
    position = prompt_count; out->target_steps = prompt_count;""")
    return before + body
