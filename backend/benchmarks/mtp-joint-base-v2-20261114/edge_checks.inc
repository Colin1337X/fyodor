/* Additional oracle comparisons for joint-base stopping and output bounds. */
static int joint_edge_case(nya_llm_context *target, nya_llm_context *assistant,
    nya_generation_request *request, size_t *failures, size_t *eos, size_t *context)
{
    nya_generation_response expected={0},actual={0};char error[512]={0};
    int reference=nya_llm_generate(target,request,&expected,error,sizeof(error));
    int candidate=nya_llm_generate_draft(target,assistant,request,&actual,error,sizeof(error));
    int bad=(reference==0)!=(candidate==0);
    if(!reference && !candidate) {
        bad=expected.generated_tokens!=actual.generated_tokens || expected.stop_reason!=actual.stop_reason ||
            expected.text_length!=actual.text_length || memcmp(expected.text,actual.text,expected.text_length)!=0;
        if(expected.stop_reason==NYA_GENERATION_STOP_EOS)++*eos;
        if(expected.stop_reason==NYA_GENERATION_STOP_CONTEXT)++*context;
    } else if(reference && candidate)++*failures;
    free(expected.text);free(actual.text);
    if(bad)fprintf(stderr,"Joint edge mismatch: limit=%zu output=%zu reference=%d candidate=%d\n",
        request->max_tokens,request->max_output_bytes,reference,candidate);
    return bad;
}
static int joint_edge_checks(nya_llm_context *target,nya_llm_context *assistant)
{
    nya_generation_request request={0};
    request.prompt="a";request.top_p=1;request.speculative_tokens=4;request.max_output_bytes=2048;
    size_t cases=0,failures=0,eos=0,context=0;
    const size_t limits[]={1,2,3,4,5,8,12,16,17};
    for(size_t i=0;i<sizeof(limits)/sizeof(limits[0]);++i) {
        request.max_tokens=limits[i];
        if(joint_edge_case(target,assistant,&request,&failures,&eos,&context))return 1;
        ++cases;
    }
    request.max_tokens=12;
    for(size_t token=0;token<8;++token) {
        target->tokenizer.stop_tokens[token]=assistant->tokenizer.stop_tokens[token]=1;
        int bad=joint_edge_case(target,assistant,&request,&failures,&eos,&context);
        target->tokenizer.stop_tokens[token]=assistant->tokenizer.stop_tokens[token]=0;
        if(bad)return 1;
        ++cases;
    }
    const size_t capacities[]={1,2,3,4,8};
    for(size_t i=0;i<sizeof(capacities)/sizeof(capacities[0]);++i) {
        request.max_output_bytes=capacities[i];
        if(joint_edge_case(target,assistant,&request,&failures,&eos,&context))return 1;
        ++cases;
    }
    if(cases!=22 || !failures || !eos || !context)return 1;
    printf("Joint edge cases: %zu; matched_errors=%zu; eos=%zu; context=%zu\n",cases,failures,eos,context);
    return 0;
}
