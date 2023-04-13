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
#define ROW 1024
#define COL 1024
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
    Matrix A(ROW,COL,0);
    Matrix B(ROW,COL,0);
    Matrix C1(ROW,COL,0);
    Matrix C2(ROW,COL,0);
    Matrix C3(ROW,COL,0);
    Matrix C4(ROW,COL,0);
    Matrix C5(ROW,COL,0);
    Matrix C6(ROW,COL,0);
    Matrix C7(ROW,COL,0);
    A.set_random(MIN,MAX);
    B.set_random(MIN,MAX);

    clock_t start;

    start=clock();
    C1=Gold(A,B);
    double sec=(clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Gold solution Consumes "<<sec<<" seconds."<<endl;
    

    start=clock();
    C2=A*B;
    sec=(clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Gold solution with omp Consumes "<<sec<<" seconds."<<endl;
    error(C1,C2);

    
    Matrix NB(B.T());
    start=clock();
    C3=Gold_with_transpose(A,NB);
    sec=(clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Gold solution with omp and transpose Consumes "<<sec<<" seconds."<<endl;
    error(C1,C3);

    start=clock();
    C4=Gold_single_packing(A,B);
    sec=(clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Gold solution single packing Consumes "<<sec<<" seconds."<<endl;
    error(C1,C4);

    start=clock();
    C5=Gold_double_packing(A,B);
    sec=(clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Gold solution double packing Consumes "<<sec<<" seconds."<<endl;
    error(C1,C5);

    start=clock();
    C6=Gold_triple_packing(A,B);
    sec=(clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Gold solution triple packing Consumes "<<sec<<" seconds."<<endl;
    error(C1,C6);

    start=clock();
    C7=Tiling(A,B);
    sec=(clock()-start)/(double)CLOCKS_PER_SEC;
    cout<<"Tiling solution Consumes "<<sec<<" seconds."<<endl;
    error(C1,C7);
    
}