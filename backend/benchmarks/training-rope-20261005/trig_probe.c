#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
extern double (*__imp_cos)(double);
extern double (*__imp_sin)(double);
int main(int argc,char **argv) {
 double angle=argc>1?strtod(argv[1],NULL):(double)FLT_MAX;
 double c=cos(angle),s=sin(angle);
 double (*volatile cosine)(double)=cos,(*volatile sine)(double)=sin;
 printf("angle=%.17g paired_cos=%.17g paired_sin=%.17g separate_cos=%.17g separate_sin=%.17g\n",angle,c,s,cosine(angle),sine(angle));
 printf("long_cos=%.17g long_sin=%.17g\n",(double)cosl((long double)angle),(double)sinl((long double)angle));
 printf("import_cos=%.17g import_sin=%.17g\n",__imp_cos(angle),__imp_sin(angle));
 return 0;
}
