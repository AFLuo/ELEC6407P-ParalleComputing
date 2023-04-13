#include<iostream>
#include<cstdlib>
#include<time.h>
#include<cmath>
#include "matrix.h"
#include<assert.h>
#include<omp.h>
using namespace std;
#define N 1024
#define MIN 1
#define MAX 100
// #define ROW 1024
// #define COL 1024
void error(Matrix a,Matrix b)
{
    double error=0.0;
    for(int i=0;i<a.rows();i++)
        for(int j=0;j<a.columns();j++)
            error+=fabs(a(i,j)-b(i,j));
    cout<<"Error: "<<error<<endl;
}
int main()
{
    

    clock_t start;
    for(int i=0;i<5;i++)
    {
        
        int ROW=(i+1)*512;
        int COL=(i+1)*512;
        cout<<"Size: "<<ROW<<" by "<<COL<<endl;

        Matrix A(ROW,COL,0);
        Matrix B(ROW,COL,0);
        Matrix C1(ROW,COL,0);
        Matrix C2(ROW,COL,0);
        Matrix C3(ROW,COL,0);
        Matrix C4(ROW,COL,0);
        Matrix C5(ROW,COL,0);
        A.set_random(MIN,MAX);
        B.set_random(MIN,MAX);

        start=clock();
        C1=A*B;
        double sec=(clock()-start)/(double)CLOCKS_PER_SEC;
        cout<<"Gold solution with omp Consumes "<<sec<<" seconds."<<endl;
        error(C1,C1);

        
        Matrix NB(B.T());
        start=clock();
        C2=Gold_with_transpose(A,NB);
        sec=(clock()-start)/(double)CLOCKS_PER_SEC;
        cout<<"Gold solution with omp and transpose Consumes "<<sec<<" seconds."<<endl;
        error(C1,C2);

        start=clock();
        C3=Gold_single_packing(A,B);
        sec=(clock()-start)/(double)CLOCKS_PER_SEC;
        cout<<"Gold solution single packing Consumes "<<sec<<" seconds."<<endl;
        error(C1,C3);

        start=clock();
        C4=Gold_double_packing(A,B);
        sec=(clock()-start)/(double)CLOCKS_PER_SEC;
        cout<<"Gold solution double packing Consumes "<<sec<<" seconds."<<endl;
        error(C1,C4);

        start=clock();
        C5=Tiling(A,B);
        sec=(clock()-start)/(double)CLOCKS_PER_SEC;
        cout<<"Tiling solution Consumes "<<sec<<" seconds."<<endl;
        error(C1,C5);
        cout<<"*******************************"<<endl;
    }
    
}