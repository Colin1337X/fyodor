#include "model.h"
#include "execution.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(void) {
 nya_model_registry *r=calloc(1,sizeof(*r));if(!r)return 2;nya_model_registry_init(r);
 const char *paths[]={"E:/models/mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf","E:/models/mtp-gemma-4-12B-it.gguf","E:/models/nomic-embed-text-v2-moe-q8_0.gguf"};
 for(unsigned i=0;i<3;++i){const nya_model *m=NULL;int result=nya_model_load(r,paths[i],&m);printf("%s\nload=%d\n",paths[i],result);
 if(m){printf("architecture=%s inference=%d generation=%d draft=%d\nexecution_error=%s\ngeneration_error=%s\n",m->architecture,m->inference_supported,m->generation_supported,m->draft_supported,m->execution_error,m->generation_error);
 if(i==0 && m->inference_supported){float *pixels=calloc(240*240*3,sizeof(float));int64_t shape[]={240,240,3};nya_execution_request req={0};nya_execution_response res={0};char error[512]={0};
 req.input_name="image_rgb";req.output_name="embeddings";req.input_type=NYA_TENSOR_FLOAT32;req.input_shape=shape;req.input_rank=3;req.input_data=pixels;req.input_data_size=240*240*3*sizeof(float);req.max_output_bytes=16*1024*1024;
 int rc=nya_execution_run((nya_model *)m,&req,&res,error,sizeof(error));size_t nonfinite=0;for(size_t j=0;j<res.output_data_size/sizeof(float);++j)if(!isfinite(((float*)res.output_data)[j]))++nonfinite;
 printf("image_projector_result=%d error=%s rank=%zu shape=%lld,%lld bytes=%zu nonfinite=%zu\n",rc,error,res.output_rank,(long long)res.output_shape[0],(long long)res.output_shape[1],res.output_data_size,nonfinite);nya_execution_response_free(&res);free(pixels);
 float audio[640]={0};int64_t audio_shape[]={1,640};req.input_name="audio_features";req.input_rank=2;req.input_shape=audio_shape;req.input_data=audio;req.input_data_size=sizeof(audio);error[0]=0;
 rc=nya_execution_run((nya_model *)m,&req,&res,error,sizeof(error));nonfinite=0;for(size_t j=0;j<res.output_data_size/sizeof(float);++j)if(!isfinite(((float*)res.output_data)[j]))++nonfinite;
 printf("audio_projector_result=%d error=%s rank=%zu shape=%lld,%lld bytes=%zu nonfinite=%zu\n",rc,error,res.output_rank,(long long)res.output_shape[0],(long long)res.output_shape[1],res.output_data_size,nonfinite);nya_execution_response_free(&res);}}
 fflush(stdout);}
 nya_model_registry_shutdown(r);free(r);nya_execution_global_shutdown();return 0;
}

