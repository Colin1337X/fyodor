from pathlib import Path

def compact_source(root):
 s=(root/'optcpp/matvec/gemma_gate.cu').read_text()
 s=s.replace('experimental packed Q4_K','experimental compact Q4_K').replace('2400','2160').replace('nya_gate_up_gemma_byte_3840','nya_gate_up_gemma_compact_3840').replace('base/256*160','base/256*144').replace('#pragma unroll 8','#pragma unroll 1').replace('g[32+j/2*32+lane]','g[16+j/2*32+lane]').replace('u[32+j/2*32+lane]','u[16+j/2*32+lane]')
 a=s.index('        const unsigned *gs32=');b=s.index('        #pragma unroll',a)
 s=s[:a]+'        const unsigned char *gs=g+4,*us=u+4;\n'+s[b:]
 s=s.replace('(G[j/4]>>((j%4)*8))&255U','j<4 ? gs[j]&63U : (gs[j+4]&15U)|((gs[j-4]>>6)<<4)').replace('(G[2+j/4]>>((j%4)*8))&255U','j<4 ? gs[j+4]&63U : (gs[j+4]>>4)|((gs[j]>>6)<<4)').replace('(U[j/4]>>((j%4)*8))&255U','j<4 ? us[j]&63U : (us[j+4]&15U)|((us[j-4]>>6)<<4)').replace('(U[2+j/4]>>((j%4)*8))&255U','j<4 ? us[j+4]&63U : (us[j+4]>>4)|((us[j]>>6)<<4)')
 return s.replace('opt-in runtime dispatch','isolated experiment; no runtime dispatch')
