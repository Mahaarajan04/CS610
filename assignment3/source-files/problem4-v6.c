// rollno-prob4-v6.c — Partial-sum reuse + loop permutation (always writes results-v6.txt)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)
#define ABS(x) ((x) < 0.0 ? -(x) : (x))

static inline void die(const char *msg){ fprintf(stderr,"%s\n",msg); exit(EXIT_FAILURE); }

int main(void){
    double a[120], b[30];

    FILE* fp = fopen("./disp.txt","r");
    if(!fp) die("Error: could not open disp.txt");
    for(int i=0;i<120;++i)
        if(fscanf(fp,"%lf",&a[i])!=1) die("Error reading disp.txt");
    fclose(fp);

    fp = fopen("./grid.txt","r");
    if(!fp) die("Error: could not open grid.txt");
    for(int i=0;i<30;++i)
        if(fscanf(fp,"%lf",&b[i])!=1) die("Error reading grid.txt");
    fclose(fp);

    const double kk = 0.3;

    double C[10][10], D[10], EY[10], E[10];
    int p = 0;
    for(int t=0;t<10;++t){
        for(int j=0;j<10;++j) C[t][j]=a[p++];
        D[t]=a[p++]; EY[t]=a[p++];
    }
    for(int t=0;t<10;++t) E[t]=kk*EY[t];

    double Xstart[10], Xend[10], Xstep[10];
    int S[10];
    for(int v=0,q=0; v<10; ++v){
        Xstart[v]=b[q++]; Xend[v]=b[q++]; Xstep[v]=b[q++];
        double span=Xend[v]-Xstart[v];
        int s=(int)floor(span/Xstep[v]);
        S[v]=(s<0)?0:s;
    }

    // sort indices by trip count ascending (smaller S outer, larger inner)
    int idx[10]; for(int i=0;i<10;++i) idx[i]=i;
    for(int i=0;i<9;++i)
        for(int j=i+1;j<10;++j)
            if(S[idx[i]]>S[idx[j]]){int tmp=idx[i]; idx[i]=idx[j]; idx[j]=tmp;}

    const int d0=idx[0], d1=idx[1], d2=idx[2], d3=idx[3], d4=idx[4];
    const int d5=idx[5], d6=idx[6], d7=idx[7], d8=idx[8], d9=idx[9];

    FILE* fptr=fopen("./results-v6.txt","w");
    if(!fptr) die("Error: cannot open results-v6.txt");

    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC_RAW,&t0);

    long pnts=0;
    double x[10];
    double part0[10],part1[10],part2[10],part3[10],part4[10],
           part5[10],part6[10],part7[10],part8[10];

    for(int t=0;t<10;++t)
        part0[t]=part1[t]=part2[t]=part3[t]=part4[t]=part5[t]=part6[t]=part7[t]=part8[t]=0.0;

    for(int r0=0; r0<S[d0]; ++r0){
        const double x0=Xstart[d0]+r0*Xstep[d0];
        x[d0]=x0;
        for(int t=0;t<10;++t) part0[t]=C[t][d0]*x0;

        for(int r1=0; r1<S[d1]; ++r1){
            const double x1=Xstart[d1]+r1*Xstep[d1];
            x[d1]=x1;
            for(int t=0;t<10;++t) part1[t]=part0[t]+C[t][d1]*x1;

            for(int r2=0; r2<S[d2]; ++r2){
                const double x2=Xstart[d2]+r2*Xstep[d2];
                x[d2]=x2;
                for(int t=0;t<10;++t) part2[t]=part1[t]+C[t][d2]*x2;

                for(int r3=0; r3<S[d3]; ++r3){
                    const double x3=Xstart[d3]+r3*Xstep[d3];
                    x[d3]=x3;
                    for(int t=0;t<10;++t) part3[t]=part2[t]+C[t][d3]*x3;

                    for(int r4=0; r4<S[d4]; ++r4){
                        const double x4=Xstart[d4]+r4*Xstep[d4];
                        x[d4]=x4;
                        for(int t=0;t<10;++t) part4[t]=part3[t]+C[t][d4]*x4;

                        for(int r5=0; r5<S[d5]; ++r5){
                            const double x5=Xstart[d5]+r5*Xstep[d5];
                            x[d5]=x5;
                            for(int t=0;t<10;++t) part5[t]=part4[t]+C[t][d5]*x5;

                            for(int r6=0; r6<S[d6]; ++r6){
                                const double x6=Xstart[d6]+r6*Xstep[d6];
                                x[d6]=x6;
                                for(int t=0;t<10;++t) part6[t]=part5[t]+C[t][d6]*x6;

                                for(int r7=0; r7<S[d7]; ++r7){
                                    const double x7=Xstart[d7]+r7*Xstep[d7];
                                    x[d7]=x7;
                                    for(int t=0;t<10;++t) part7[t]=part6[t]+C[t][d7]*x7;

                                    for(int r8=0; r8<S[d8]; ++r8){
                                        const double x8=Xstart[d8]+r8*Xstep[d8];
                                        x[d8]=x8;
                                        for(int t=0;t<10;++t) part8[t]=part7[t]+C[t][d8]*x8;

                                        for(int r9=0; r9<S[d9]; ++r9){
                                            const double x9=Xstart[d9]+r9*Xstep[d9];
                                            x[d9]=x9;

                                            int ok=1;
                                            for(int t=0;t<10;++t){
                                                double sum=part8[t]+C[t][d9]*x9;
                                                double q=ABS(sum-D[t]);
                                                if(q>E[t]){ok=0;break;}
                                            }

                                            if(ok){
                                                ++pnts;
                                                fprintf(fptr,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                                                    x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]);
                                            }
                                        } // r9
                                    } // r8
                                } // r7
                            } // r6
                        } // r5
                    } // r4
                } // r3
            } // r2
        } // r1
    } // r0

    fclose(fptr);

    clock_gettime(CLOCK_MONOTONIC_RAW,&t1);
    double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/NSEC_SEC_MUL;
    printf("result pnts: %ld\n",pnts);
    printf("Total time = %f seconds\n",sec);
    return 0;
}
