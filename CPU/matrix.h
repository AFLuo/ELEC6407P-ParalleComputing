#include<iostream>
#include<cstdlib>
#include<immintrin.h>
#include<time.h>
#include<omp.h>
#include<random>
using namespace std;
class Matrix
{
    private:
    int row, col;
    double** matrix;

    public:
    //class definition
    //constructor and de-constructor
    Matrix():matrix(nullptr),row(0),col(0){}
    Matrix(int r,int c):row(r),col(c)
    {
        row=r;
        col=c;
        matrix=(double**)malloc(r*sizeof(double*));
        double **end,**next;
        next=matrix;
        end=matrix+r;
        while(next!=end)
        {
            *next=(double*)malloc(c*sizeof(double));
            next++;
        }
    }
    Matrix(int r,int c,double init):row(r),col(c){
        matrix=(double**)malloc(r*sizeof(double*));
        double **end,**next,*it,*p;
        next=matrix;
        end=matrix+r;
        while(next!=end)
        {
            it=*next=(double*)malloc(c*sizeof(double));
            p=it+c;
            while(it!=p)
            {
                *it=init;
                it++;
            }
            next++;
        }
    }
    Matrix(const Matrix& a)
    {
        row=a.row;
        col=a.col;

        matrix=(double**)malloc(row*sizeof(double*));
        double **end_a,**next_a,*it_a,*p_a;
        double **end_b,**next_b,*it_b,*p_b;
        
        next_a=matrix;
        end_a=matrix+row;
        next_b=a.matrix;
        end_b=next_b+row;

        while(next_a!=end_a)
        {
            it_a=*next_a=(double*)malloc(col*sizeof(double));
            p_a=it_a+col;
            it_b=*next_b;
            p_b=it_b+col;
            while(it_a!=p_a)
            {
                *it_a=*it_b;
                it_a++;
                it_b++;
            }
            next_a++;
            next_b++;
        }
    }
    virtual ~Matrix(){
        if(!matrix) return;
        double **next,**end;
        next=matrix;
        end=matrix+row;
        do
        {
            free(*next);
            next++;
        }while(next!=end);
        
        row=col=0;
        free(matrix);
    }


        
    //operators
    void operator=(const Matrix& b){
        double **next_a,**end_a,*it_a,*p_a;
        double **next_b,**end_b,*it_b,*p_b;
        next_a=matrix;
        end_a=matrix+row;
        next_b=b.matrix;
        end_b=b.matrix+b.row;

        while(next_a!=end_a)
        {
            it_a=*next_a;
            p_a=it_a+col;
            it_b=*next_b;
            p_b=it_b+col;
            while(it_a!=p_a)
            {
                *it_a=*it_b;
                it_a++,it_b++;
            }
            next_a++,next_b++;
        }
        
    }
    void operator=(double scalar)
    {
        double **next,**end,*it,*p;
        next=matrix;
        end=matrix+row;
        while(next!=end)
        {
            it=*next;
            p=it+col;
            while(it!=p)
            {
                *it=scalar;
                it++;
            }
            next++;
        }
    }
    Matrix operator+(double scalar)
    {
        double **next,**end,*it,*p;
        next=matrix;
        end=matrix+row;
        while(next!=end)
        {
            it=*next;
            p=it+col;
            while(it!=p)
            {
                *it+=scalar;
                it++;
            }
            next++;
        }
        return *this;
    }
    Matrix operator+(const Matrix b){
        if(row!=b.row || col!=b.col)
        {
            printf("doublehe dimension does not match, nothing happened.");
            return *this;
        }
        //先初始化一个全为0的矩阵
        Matrix c(row,col,0);
        double **next_a,**end_a,*it_a,*p_a;
        double **next_b,**end_b,*it_b,*p_b;
        double **next_c,**end_c,*it_c,*p_c;
        next_a=matrix;
        end_a=matrix+row;
        next_b=b.matrix;
        end_b=b.matrix+b.row;
        next_c=c.matrix;
        end_c=c.matrix+c.row;
        //迭代相加
        while(next_a!=end_a)
        {
            it_a=*next_a;
            p_a=it_a+col;
            it_b=*next_b;
            p_b=it_b+col;
            it_c=*next_c;
            p_c=it_c+col;

            while(it_a!=p_a)
            {
                *it_c=*it_a+*it_b;
                it_a++;
                it_b++;
                it_c++;
            }
            next_a++;
            next_b++;
            next_c++;
        }
        return c;
    }
    Matrix operator*(double scalar)
    {
        double **next,**end,*it,*p;
        next=matrix;
        end=matrix+row;
        while(next!=end)
        {
            it=*next;
            p=it+col;
            while(it!=p)
            {
                *it=(*it)*scalar;
                it++;
            }
            next++;
        }
        return *this;
    }
    double& operator()(int i,int j)
    {
        //& represents reference that you can modify the value
        return matrix[i][j];
    }
    
    //attributes
    int rows()
    {
        return row;
    }  
    int columns()
    {
        return col;
    }
    Matrix getColumn(int i)
    {
        Matrix tmp(row,1,0);
        for(int j=0;j<row;j++)
        {
            tmp(j,0)=matrix[j][i];
        }
        return tmp;
    }
    Matrix getRow(int i)
    {
        Matrix tmp(1,col,0);
        for(int j=0;j<col;j++)
        {
            tmp(0,j)=matrix[i][j];
        }
        return tmp;
    }
    Matrix getDiagonal()
    {
        int size=row>col?row:col;
        Matrix tmp(1,size,0);
        for(int j=0;j<col;j++)
        {
            tmp(0,j)=matrix[j][j];
        }
        return tmp;
    }
    
    //some special functions
    void eye()
    {
        //transform this into a unit matrix
        int n=row<col?row:col;
        for(int i=0;i<n;i++)
            matrix[i][i]=1;
    }
    Matrix T()
    {
        int ra=col,ca=row;
        Matrix c(ra,ca,0);
        for(int i=0;i<ra;i++)
            for(int j=0;j<ca;j++)
                c(i,j)=(*this)(j,i);
        return c;
    }
    void set_random(double min, double max)
    {  
        random_device rd;
        default_random_engine eng(rd());
        uniform_real_distribution<double> distri(min,max);
        
        //we should claim i,j is private so that i,j won't be modified
        int i=0,j=0;
        #pragma omp for private(i,j)
        for(i=0;i<row;i++)
            for(j=0;j<col;j++)
                matrix[i][j]=distri(eng);
    }
};
Matrix Gold(Matrix a,Matrix b)
{
    if(a.columns()!=b.rows())
    {
        printf("doublehe dimension does not match, nothing happened.");
        return a;
    }
    Matrix c(a.rows(),b.columns(),0);
    
    int i=0,j=0,k=0;
    for(int i=0;i<a.rows();i++)
        for(int j=0;j<b.columns();j++)
            for (int k=0;k<a.columns();k++)
                c(i,j)+=a(i,k)*b(k,j);
    return c;
}
Matrix operator*(Matrix a,Matrix b)
{
    if(a.columns()!=b.rows())
    {
        printf("doublehe dimension does not match, nothing happened.");
        return a;
    }
    Matrix c(a.rows(),b.columns(),0);
    
    int i=0,j=0,k=0;
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<a.rows();i++)
        for(int j=0;j<b.columns();j++)
        {
            double temp_c=0.0;
            for (int k=0;k<a.columns();k++)
                temp_c+=a(i,k)*b(k,j);
            c(i,j)=temp_c;
        }
    return c;
}
Matrix Gold_with_transpose(Matrix a, Matrix b)
{
    if(a.columns()!=b.columns())
    {
        printf("doublehe dimension does not match, nothing happened.");
        return a;
    }
    Matrix c(a.rows(),b.rows(),0);
    
    int i=0,j=0,k=0;
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<a.rows();i++)
        for(int j=0;j<b.rows();j++)
        {
            double temp_c=0.0;
            for (int k=0;k<a.columns();k++)
            {   
                temp_c+=a(i,k)*b(j,k);
            }
            c(i,j)=temp_c;
        }
            
    return c;
}
Matrix Gold_single_packing(Matrix a, Matrix b)
{
    if(a.columns()!=b.rows())
    {
        printf("doublehe dimension does not match, nothing happened.");
        return a;
    }
    Matrix c(a.rows(),b.columns(),0);
    
    int i=0,j=0,k=0;
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<a.rows();i+=4)
        for(int j=0;j<b.columns();j++)
        {
            double temp_c[4]={0};
            for (int k=0;k<a.columns();k++)
            {
                temp_c[0]+=a(i+0,k)*b(k,j);
                temp_c[1]+=a(i+1,k)*b(k,j);
                temp_c[2]+=a(i+2,k)*b(k,j);
                temp_c[3]+=a(i+3,k)*b(k,j);
            }
            c(i+0,j)=temp_c[0];
            c(i+1,j)=temp_c[1];
            c(i+2,j)=temp_c[2];
            c(i+3,j)=temp_c[3];
        }
            
    return c;
}
Matrix Gold_double_packing(Matrix a, Matrix b)
{
    if(a.columns()!=b.rows())
    {
        printf("doublehe dimension does not match, nothing happened.");
        return a;
    }
    Matrix c(a.rows(),b.columns(),0);
    
    int i=0,j=0,k=0;
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<a.rows();i+=4)
        for(int j=0;j<b.columns();j+=4)
        {   
            double temp_c[4][4]={{0}};
            for (int k=0;k<a.columns();k++)
            {
                
                temp_c[0][0]+=a(i+0,k)*b(k,j+0);
                temp_c[0][1]+=a(i+0,k)*b(k,j+1);
                temp_c[0][2]+=a(i+0,k)*b(k,j+2);
                temp_c[0][3]+=a(i+0,k)*b(k,j+3);
                
                temp_c[1][0]+=a(i+1,k)*b(k,j+0);
                temp_c[1][1]+=a(i+1,k)*b(k,j+1);
                temp_c[1][2]+=a(i+1,k)*b(k,j+2);
                temp_c[1][3]+=a(i+1,k)*b(k,j+3);

                temp_c[2][0]+=a(i+2,k)*b(k,j+0);
                temp_c[2][1]+=a(i+2,k)*b(k,j+1);
                temp_c[2][2]+=a(i+2,k)*b(k,j+2);
                temp_c[2][3]+=a(i+2,k)*b(k,j+3);

                temp_c[3][0]+=a(i+3,k)*b(k,j+0);
                temp_c[3][1]+=a(i+3,k)*b(k,j+1);
                temp_c[3][2]+=a(i+3,k)*b(k,j+2);
                temp_c[3][3]+=a(i+3,k)*b(k,j+3);
            }
            c(i+0,j+0)=temp_c[0][0];
            c(i+0,j+1)=temp_c[0][1];
            c(i+0,j+2)=temp_c[0][2];
            c(i+0,j+3)=temp_c[0][3];

            c(i+1,j+0)=temp_c[1][0];
            c(i+1,j+1)=temp_c[1][1];
            c(i+1,j+2)=temp_c[1][2];
            c(i+1,j+3)=temp_c[1][3];

            c(i+2,j+0)=temp_c[2][0];
            c(i+2,j+1)=temp_c[2][1];
            c(i+2,j+2)=temp_c[2][2];
            c(i+2,j+3)=temp_c[2][3];

            c(i+3,j+0)=temp_c[3][0];
            c(i+3,j+1)=temp_c[3][1];
            c(i+3,j+2)=temp_c[3][2];
            c(i+3,j+3)=temp_c[3][3];
        }
            
    return c;
}

Matrix Gold_triple_packing(Matrix a, Matrix b)
{
    if(a.columns()!=b.rows())
    {
        printf("doublehe dimension does not match, nothing happened.");
        return a;
    }
    Matrix c(a.rows(),b.columns(),0);
    
    int i=0,j=0,k=0;
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<a.rows();i+=4)
        for(int j=0;j<b.columns();j+=4)
        {
            double temp_c[4][4]={{0}};
            for (int k=0;k<a.columns();k+=4)
            {
                /****************************/
                temp_c[0][0]+=a(i+0,k+0)*b(k+0,j+0);
                temp_c[0][1]+=a(i+0,k+0)*b(k+0,j+1);
                temp_c[0][2]+=a(i+0,k+0)*b(k+0,j+2);
                temp_c[0][3]+=a(i+0,k+0)*b(k+0,j+3);
                
                temp_c[1][0]+=a(i+1,k+0)*b(k+0,j+0);
                temp_c[1][1]+=a(i+1,k+0)*b(k+0,j+1);
                temp_c[1][2]+=a(i+1,k+0)*b(k+0,j+2);
                temp_c[1][3]+=a(i+1,k+0)*b(k+0,j+3);

                temp_c[2][0]+=a(i+2,k+0)*b(k+0,j+0);
                temp_c[2][1]+=a(i+2,k+0)*b(k+0,j+1);
                temp_c[2][2]+=a(i+2,k+0)*b(k+0,j+2);
                temp_c[2][3]+=a(i+2,k+0)*b(k+0,j+3);

                temp_c[3][0]+=a(i+3,k+0)*b(k+0,j+0);
                temp_c[3][1]+=a(i+3,k+0)*b(k+0,j+1);
                temp_c[3][2]+=a(i+3,k+0)*b(k+0,j+2);
                temp_c[3][3]+=a(i+3,k+0)*b(k+0,j+3);

                /****************************/
                temp_c[0][0]+=a(i+0,k+1)*b(k+1,j+0);
                temp_c[0][1]+=a(i+0,k+1)*b(k+1,j+1);
                temp_c[0][2]+=a(i+0,k+1)*b(k+1,j+2);
                temp_c[0][3]+=a(i+0,k+1)*b(k+1,j+3);
                
                temp_c[1][0]+=a(i+1,k+1)*b(k+1,j+0);
                temp_c[1][1]+=a(i+1,k+1)*b(k+1,j+1);
                temp_c[1][2]+=a(i+1,k+1)*b(k+1,j+2);
                temp_c[1][3]+=a(i+1,k+1)*b(k+1,j+3);

                temp_c[2][0]+=a(i+2,k+1)*b(k+1,j+0);
                temp_c[2][1]+=a(i+2,k+1)*b(k+1,j+1);
                temp_c[2][2]+=a(i+2,k+1)*b(k+1,j+2);
                temp_c[2][3]+=a(i+2,k+1)*b(k+1,j+3);

                temp_c[3][0]+=a(i+3,k+1)*b(k+1,j+0);
                temp_c[3][1]+=a(i+3,k+1)*b(k+1,j+1);
                temp_c[3][2]+=a(i+3,k+1)*b(k+1,j+2);
                temp_c[3][3]+=a(i+3,k+1)*b(k+1,j+3);
                
                /****************************/
                temp_c[0][0]+=a(i+0,k+2)*b(k+2,j+0);
                temp_c[0][1]+=a(i+0,k+2)*b(k+2,j+1);
                temp_c[0][2]+=a(i+0,k+2)*b(k+2,j+2);
                temp_c[0][3]+=a(i+0,k+2)*b(k+2,j+3);
                
                temp_c[1][0]+=a(i+1,k+2)*b(k+2,j+0);
                temp_c[1][1]+=a(i+1,k+2)*b(k+2,j+1);
                temp_c[1][2]+=a(i+1,k+2)*b(k+2,j+2);
                temp_c[1][3]+=a(i+1,k+2)*b(k+2,j+3);

                temp_c[2][0]+=a(i+2,k+2)*b(k+2,j+0);
                temp_c[2][1]+=a(i+2,k+2)*b(k+2,j+1);
                temp_c[2][2]+=a(i+2,k+2)*b(k+2,j+2);
                temp_c[2][3]+=a(i+2,k+2)*b(k+2,j+3);

                temp_c[3][0]+=a(i+3,k+2)*b(k+2,j+0);
                temp_c[3][1]+=a(i+3,k+2)*b(k+2,j+1);
                temp_c[3][2]+=a(i+3,k+2)*b(k+2,j+2);
                temp_c[3][3]+=a(i+3,k+2)*b(k+2,j+3);
                
                /****************************/
                temp_c[0][0]+=a(i+0,k+3)*b(k+3,j+0);
                temp_c[0][1]+=a(i+0,k+3)*b(k+3,j+1);
                temp_c[0][2]+=a(i+0,k+3)*b(k+3,j+2);
                temp_c[0][3]+=a(i+0,k+3)*b(k+3,j+3);
                
                temp_c[1][0]+=a(i+1,k+3)*b(k+3,j+0);
                temp_c[1][1]+=a(i+1,k+3)*b(k+3,j+1);
                temp_c[1][2]+=a(i+1,k+3)*b(k+3,j+2);
                temp_c[1][3]+=a(i+1,k+3)*b(k+3,j+3);

                temp_c[2][0]+=a(i+2,k+3)*b(k+3,j+0);
                temp_c[2][1]+=a(i+2,k+3)*b(k+3,j+1);
                temp_c[2][2]+=a(i+2,k+3)*b(k+3,j+2);
                temp_c[2][3]+=a(i+2,k+3)*b(k+3,j+3);

                temp_c[3][0]+=a(i+3,k+3)*b(k+3,j+0);
                temp_c[3][1]+=a(i+3,k+3)*b(k+3,j+1);
                temp_c[3][2]+=a(i+3,k+3)*b(k+3,j+2);
                temp_c[3][3]+=a(i+3,k+3)*b(k+3,j+3);
            }
            c(i+0,j+0)=temp_c[0][0];
            c(i+0,j+1)=temp_c[0][1];
            c(i+0,j+2)=temp_c[0][2];
            c(i+0,j+3)=temp_c[0][3];

            c(i+1,j+0)=temp_c[1][0];
            c(i+1,j+1)=temp_c[1][1];
            c(i+1,j+2)=temp_c[1][2];
            c(i+1,j+3)=temp_c[1][3];

            c(i+2,j+0)=temp_c[2][0];
            c(i+2,j+1)=temp_c[2][1];
            c(i+2,j+2)=temp_c[2][2];
            c(i+2,j+3)=temp_c[2][3];

            c(i+3,j+0)=temp_c[3][0];
            c(i+3,j+1)=temp_c[3][1];
            c(i+3,j+2)=temp_c[3][2];
            c(i+3,j+3)=temp_c[3][3];
        }
            
    return c;
}
Matrix Tiling(Matrix a, Matrix b)
{
    if(a.columns()!=b.rows())
    {
        printf("doublehe dimension does not match, nothing happened.");
        return a;
    }
    Matrix c(a.rows(),b.columns(),0);

    int tile_size=4;
    // int i=0,j=0,k=0;
    #pragma omp parallel for 
    for(int i=0;i<a.rows();i+=tile_size)
        for(int j=0;j<b.columns();j+=tile_size)
        {
            double na[4][4]={{0}};
            double nb[4][4]={{0}};
            double nc[4][4]={{0}};
            for(int k=0;k<a.columns();k+=tile_size)
            {
                //load submatrix of a into caches
                na[0][0]=a(i+0,k+0);
                na[0][1]=a(i+0,k+1);
                na[0][2]=a(i+0,k+2);
                na[0][3]=a(i+0,k+3);

                na[1][0]=a(i+1,k+0);
                na[1][1]=a(i+1,k+1);
                na[1][2]=a(i+1,k+2);
                na[1][3]=a(i+1,k+3);

                na[2][0]=a(i+2,k+0);
                na[2][1]=a(i+2,k+1);
                na[2][2]=a(i+2,k+2);
                na[2][3]=a(i+2,k+3);

                na[3][0]=a(i+3,k+0);
                na[3][1]=a(i+3,k+1);
                na[3][2]=a(i+3,k+2);
                na[3][3]=a(i+3,k+3);

                //load submatrix of b into caches
                nb[0][0]=b(k+0,j+0);
                nb[0][1]=b(k+0,j+1);
                nb[0][2]=b(k+0,j+2);
                nb[0][3]=b(k+0,j+3);

                nb[1][0]=b(k+1,j+0);
                nb[1][1]=b(k+1,j+1);
                nb[1][2]=b(k+1,j+2);
                nb[1][3]=b(k+1,j+3);

                nb[2][0]=b(k+2,j+0);
                nb[2][1]=b(k+2,j+1);
                nb[2][2]=b(k+2,j+2);
                nb[2][3]=b(k+2,j+3);

                nb[3][0]=b(k+3,j+0);
                nb[3][1]=b(k+3,j+1);
                nb[3][2]=b(k+3,j+2);
                nb[3][3]=b(k+3,j+3);
                
                //matrix multiplication of sub-matrices
                nc[0][0]+=na[0][0]*nb[0][0]+na[0][1]*nb[1][0]+na[0][2]*nb[2][0]+na[0][3]*nb[3][0];
                nc[0][1]+=na[0][0]*nb[0][1]+na[0][1]*nb[1][1]+na[0][2]*nb[2][1]+na[0][3]*nb[3][1];
                nc[0][2]+=na[0][0]*nb[0][2]+na[0][1]*nb[1][2]+na[0][2]*nb[2][2]+na[0][3]*nb[3][2];
                nc[0][3]+=na[0][0]*nb[0][3]+na[0][1]*nb[1][3]+na[0][2]*nb[2][3]+na[0][3]*nb[3][3];

                nc[1][0]+=na[1][0]*nb[0][0]+na[1][1]*nb[1][0]+na[1][2]*nb[2][0]+na[1][3]*nb[3][0];
                nc[1][1]+=na[1][0]*nb[0][1]+na[1][1]*nb[1][1]+na[1][2]*nb[2][1]+na[1][3]*nb[3][1];
                nc[1][2]+=na[1][0]*nb[0][2]+na[1][1]*nb[1][2]+na[1][2]*nb[2][2]+na[1][3]*nb[3][2];
                nc[1][3]+=na[1][0]*nb[0][3]+na[1][1]*nb[1][3]+na[1][2]*nb[2][3]+na[1][3]*nb[3][3];

                nc[2][0]+=na[2][0]*nb[0][0]+na[2][1]*nb[1][0]+na[2][2]*nb[2][0]+na[2][3]*nb[3][0];
                nc[2][1]+=na[2][0]*nb[0][1]+na[2][1]*nb[1][1]+na[2][2]*nb[2][1]+na[2][3]*nb[3][1];
                nc[2][2]+=na[2][0]*nb[0][2]+na[2][1]*nb[1][2]+na[2][2]*nb[2][2]+na[2][3]*nb[3][2];
                nc[2][3]+=na[2][0]*nb[0][3]+na[2][1]*nb[1][3]+na[2][2]*nb[2][3]+na[2][3]*nb[3][3];

                nc[3][0]+=na[3][0]*nb[0][0]+na[3][1]*nb[1][0]+na[3][2]*nb[2][0]+na[3][3]*nb[3][0];
                nc[3][1]+=na[3][0]*nb[0][1]+na[3][1]*nb[1][1]+na[3][2]*nb[2][1]+na[3][3]*nb[3][1];
                nc[3][2]+=na[3][0]*nb[0][2]+na[3][1]*nb[1][2]+na[3][2]*nb[2][2]+na[3][3]*nb[3][2];
                nc[3][3]+=na[3][0]*nb[0][3]+na[3][1]*nb[1][3]+na[3][2]*nb[2][3]+na[3][3]*nb[3][3];
            }
            c(i+0,j+0)=nc[0][0];
            c(i+0,j+1)=nc[0][1];
            c(i+0,j+2)=nc[0][2];
            c(i+0,j+3)=nc[0][3];

            c(i+1,j+0)=nc[1][0];
            c(i+1,j+1)=nc[1][1];
            c(i+1,j+2)=nc[1][2];
            c(i+1,j+3)=nc[1][3];

            c(i+2,j+0)=nc[2][0];
            c(i+2,j+1)=nc[2][1];
            c(i+2,j+2)=nc[2][2];
            c(i+2,j+3)=nc[2][3];

            c(i+3,j+0)=nc[3][0];
            c(i+3,j+1)=nc[3][1];
            c(i+3,j+2)=nc[3][2];
            c(i+3,j+3)=nc[3][3];
        }
        return c;
}