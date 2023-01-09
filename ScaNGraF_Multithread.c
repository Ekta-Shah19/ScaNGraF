#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include<sys/stat.h>

typedef double TYPE;
#define MAX_DIM 8000*8000

void convert(TYPE** matrixA, TYPE** matrixB, int dimension);
double compute_powers(double **,double **,int,int);
double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double update_networks_multiple_iter(double ***,double ***,double ***,int,int,int,int);
double update_networks_single_iter(double ***,double ***,double ***,int,int,int);
void dominateset(double ***,double ***,int ,int ,int ,int);
void normalize3D(double ***,int ,int ,int );
void normalize2D(double **,int,int);
double ***scale3D(double ***,int , int ,int );
void symmetricity(double ***,int,int,int);
double ***allocate3D(int,int,int);
double **allocate2D(int,int);
void deallocate2D(double **,int);
void deallocate3D(double ***,int,int);

// 1 Dimensional matrix on stack
TYPE flatA[MAX_DIM];
TYPE flatB[MAX_DIM];

int main(int argc,char *argv[])
{
	int n,i,j,k,t;
	int p,q;
	int row, col;
	int flag,knn;
	int diff;
	double opmLatency;
	double **temp;
	double ***mat;
	double ***sparse_mat;
	double ***updated_mat;
	char file[1000];
	char comand[1000];
	char product[1000];
    struct stat st;
    time_t start, stop;
	FILE *fp;
	
	if(argc==1)
	{
		printf("Enter the following information:\n");
		printf("1. Number of Similarity Files(n)\n");
		printf("2. Number of Iterations\n");
		printf("3. Number of neighbors\n");
		printf("4. n Similarity Matrices\n");
		printf("5. Output Directory\n");
		exit(0);
	}
	
	time(&start);
	n=atoi(argv[1]);
	t=atoi(argv[2]);
	knn=atoi(argv[3]);
	
	for(i=0;i<n;i++)
	{
		sprintf(comand,"wc -l %s > temp.txt",argv[i+4]);
		system(comand);
	
		fp=fopen("temp.txt","r");
		fscanf(fp,"%d",&row);
		fclose(fp);
	
		col=row;
		printf("Dimension of Matrix %d: %d x %d\n",i+1,row,col);
	}
	
	mat=allocate3D(n,row,col);
	for(i=0;i<n;i++)
	{
		fp=fopen(argv[i+4],"r");
		for(j=0;j<row;j++)
			for(k=0;k<col;k++)
				fscanf(fp,"%lf\t",&mat[i][j][k]);
		fclose(fp);
	}
	
	mat=scale3D(mat,n,row,col);
		
	flag=0;
	for(i=0;i<n;i++)
		for(j=0;j<row;j++)
			if(mat[i][j][j]!=0.5)
			{
				flag=1;
				break;
			}
	
	if(flag==1)
	{
		normalize3D(mat,n,row,col);
		symmetricity(mat,n,row,col);
		printf("Networks are now Symmetric and Normalized\n");
	}
	
	printf("Considering kernel with %d neighbors\n",knn);
	
	sparse_mat=allocate3D(n,row,col);
	dominateset(sparse_mat,mat,knn,n,row,col);
	printf("Computing Local Affinity Matrix\n");
	
	printf("No. of iterations: %d\n",t);
	
	if(t==0)
	{
		printf("Zero iterations selected\n");
		printf("Program exiting\n");
		exit(0);
	}
	
	updated_mat=allocate3D(n,row,col);
	if(t==1)
		opmLatency=update_networks_single_iter(mat,sparse_mat,updated_mat,n,row,col);
	if(t>1)
		opmLatency=update_networks_multiple_iter(mat,sparse_mat,updated_mat,t,n,row,col);
	printf("Total Latency: %lf\n",opmLatency);
	
	temp=allocate2D(row,col);
	for(j=0;j<row;j++)
	{
		for(k=0;k<col;k++)
		{
			for(i=0;i<n;i++)
				temp[j][k]+=updated_mat[i][j][k];
			temp[j][k]=temp[j][k]/(double)n;
		}
	}
	printf("Averaging of diffused matrices done\n");
	
	normalize2D(temp,row,col);
	for(j=0;j<row;j++)
	{
		for(k=j;k<col;k++)
		{
			temp[j][k]=(temp[j][k]+temp[k][j])/(double)2;
			temp[k][j]=temp[j][k];
		}
	}
	printf("Matrices are normalized and symmetricized\n");
	
	deallocate3D(updated_mat,n,row);
	deallocate3D(sparse_mat,n,row);
	deallocate3D(mat,n,row);
	
	sprintf(file,"%s/FUSION_K_%d_ITR_%d.txt",argv[n+4],knn,t);
	fp=fopen(file,"w");
	for(i=0;i<row;i++)
	{
		for(j=0;j<col-1;j++)
			fprintf(fp,"%.17lf\t",temp[i][j]);
		fprintf(fp,"%.17lf\n",temp[i][j]);
	}
	fclose(fp);
	printf("Matrix written to file\n");
	
	time(&stop);
	printf("Time Taken: %lf secs\n",difftime(stop,start));
	printf("Diffusion Completed\n");
	
	sprintf(file,"%s/PARALLEL_EXECUTION_TIME.txt",argv[n+4]);
	fp=fopen(file,"a");
	fprintf(fp,"K=%d\tT=%d\tTime=%lf secs\n",knn,t,difftime(stop,start));
	fclose(fp);
	
	deallocate2D(temp,row);
	
	return 0;
}

double update_networks_multiple_iter(double ***mat,double ***sparse_mat,double ***updated_mat,int iterations,int n,int row,int col)
{
	int i,j,k,l;
	int powers;
	double **product;
	double **transpose;
	double **final;
	double **product_powers;
	double **sparse_product;
	double latency;
	double total_latency;
	
	total_latency=0.0;
	if(iterations%2==0)
		powers=iterations/2;
	else
		powers=(iterations-1)/2;
	
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			if(i!=j)
			{
				sparse_product=allocate2D(row,col);
				final=allocate2D(row,col);

				latency=optimizedParallelMultiply(sparse_mat[i],sparse_mat[j],sparse_product,row);
				printf("Time taken for Sparse %d * Sparse %d = %lf\n",i+1,j+1,latency);
				total_latency+=latency;
				
				if(powers>=2)
				{
					product_powers=allocate2D(row,col);
					
					latency=compute_powers(sparse_product,product_powers,row,powers);
					total_latency+=latency;
					printf("Time Taken to Compute (Sparse_%d * Sparse_%d)^%d = %lf\n", i+1, j+1, powers, latency);
					
					if(iterations%2==1)
					{
						latency=optimizedParallelMultiply(product_powers,sparse_mat[i],final,row);
						total_latency+=latency;
						printf("Time Taken for Performing Additional Multiplication for Odd Iterations = %lf\n",latency);
					}
					else
						for(k=0;k<row;k++)
							for(l=0;l<row;l++)
								final[k][l]=product_powers[k][l];
					deallocate2D(product_powers,row);
				}
				else
				{
					printf("Working for Power= %d\n",powers);
					if(iterations%2==1)
					{
						latency=optimizedParallelMultiply(sparse_product,sparse_mat[i],final,row);
						total_latency+=latency;
						printf("Time Taken for Multiplying Sparse %d_%d with Sparse %d = %lf\n", i+1,j+1,i+1,latency);
					}
					else
						for(k=0;k<row;k++)
							for(l=0;l<row;l++)
								final[k][l]=sparse_product[k][l];
				}
				deallocate2D(sparse_product,row);
				
				transpose=allocate2D(row,col);
				for(k=0;k<row;k++)
					for(l=0;l<row;l++)
						transpose[k][l]=final[l][k];
				
				product=allocate2D(row,col);
				latency=optimizedParallelMultiply(final,mat[j],product,row);
				total_latency+=latency;
				printf("Time Taken for Phase 1 Multiplication = %lf\n", latency);
				
				latency=optimizedParallelMultiply(product,transpose,final,row);
				total_latency+=latency;
				printf("Time Taken for Phase 2 Multiplication = %lf\n", latency);
				
				for(k=0;k<row;k++)
					for(l=0;l<row;l++)
						updated_mat[i][k][l]+=final[k][l];
				printf("Updation of View %d using View %d Completed\n",i+1,j+1);

				deallocate2D(final,row);
				deallocate2D(product,row);
				deallocate2D(transpose,row);
			}
		}
	}
	
	if(n>2)
	{
		for(i=0;i<n;i++)
			for(j=0;j<row;j++)
				for(k=0;k<row;k++)
					updated_mat[i][j][k]=updated_mat[i][j][k]/(double)(n-1);
		printf("Updated Views Averaging Complete\n");
	}
					
	printf("Total time taken for computing product and powers: %lf\n",total_latency);
	
	return total_latency;
}

double compute_powers(double **sparse_product,double **final,int row,int powers)
{
	int i,j,k,p,l;
	int diff;
	int *powrs;
	double ***product_powers;
	double latency;
	double total_latency;
	
	powrs=(int *)calloc(powers,sizeof(int));
	product_powers=allocate3D(powers,row,row);
	
	powrs[0]=1;
	for(j=0;j<row;j++)
		for(k=0;k<row;k++)
			product_powers[0][j][k]=sparse_product[j][k];
	
	i=2;p=1;
	total_latency=0.0;
	while(i<=powers)
	{
		latency=optimizedParallelMultiply(product_powers[p-1],product_powers[p-1],final,row);
		printf("Index: %d\tPower=%d\tTime Taken=%lf\n",p,i,latency);
		
		for(j=0;j<row;j++)
			for(k=0;k<row;k++)
				product_powers[p][j][k]=final[j][k];
		total_latency+=latency;
		
		powrs[p]=i;
		p++;
		i=2*i;
	}
	i=i/2;
	
	diff=powers-i;
	if(diff==1)
	{
		latency=optimizedParallelMultiply(product_powers[p-1],sparse_product,final,row);
		printf("Power=%d\tTime Taken=%lf\n",i+1,latency);
		total_latency+=latency;
		diff--;
	}
	else
	{	
		while(diff>0)
		{
			printf("Working for Power Difference=%d\n",diff);
			j=(floor)(log(diff)/(double)log(2));
			printf("Accessing Index: %d\tPower: %d\n",j,powrs[j]);
			
			latency=optimizedParallelMultiply(product_powers[p-1],product_powers[j],final,row);
			total_latency+=latency;
			printf("Multiplied Power %d to %d\n",powrs[p-1],powrs[j]);
			
			for(l=0;l<row;l++)
				for(k=0;k<row;k++)
					product_powers[p][l][k]=final[l][k];
			powrs[p]=powrs[j]+powrs[p-1];
			
			printf("Power=%d\t Time=%lf\n",powrs[p],latency);
			
			diff=diff-powrs[j];
			p++;
		}
	}
	
	free(powrs);
	deallocate3D(product_powers,powers,row);
	
	return total_latency;
}

double update_networks_single_iter(double ***mat,double ***sparse_mat,double ***updated_mat,int n,int row,int col)
{
	int i,j,k,l;
	int count;
	double **temp;
	double **transpose;
	double ***fused;
	double latency;
	double total_latency;
	
	total_latency=0;
	for(i=0;i<n;i++)
	{
		count=0;
		fused=allocate3D(n-1,row,col);
		for(j=0;j<n;j++)
			if(i!=j)
			{
				temp=allocate2D(row,col);
				latency=optimizedParallelMultiply(sparse_mat[i],mat[j],temp,row);
				printf("Time taken for Sparse Matrix %d * Matrix %d = %lf\n",i+1, j+1, latency);
				total_latency+=latency;
				
				transpose=allocate2D(row,col);
				for(k=0;k<row;k++)
					for(l=0;l<row;l++)
						transpose[k][l]=sparse_mat[i][l][k];
				
				latency=optimizedParallelMultiply(temp,transpose,fused[count],row);
				printf("Time taken for Matrix * Tr(Sparse Matrix %d) = %lf\n",i+1, latency);
				total_latency+=latency;
				
				deallocate2D(temp,row);
				deallocate2D(transpose,row);
				
				count++;
			}
		
		if(n>2)
		{
			for(j=0;j<row;j++)
				for(l=0;l<col;l++)
				{
					for(k=0;k<n-1;k++)
						updated_mat[i][j][l]+=fused[k][j][l];
					updated_mat[i][j][l]=updated_mat[i][j][l]/(double)(n-1);
				}
		}
		else
			for(j=0;j<row;j++)
				for(l=0;l<col;l++)
					updated_mat[i][j][l]=fused[0][j][l];
					
		deallocate3D(fused,n-1,row);
	}
	
	return total_latency;
}

double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){
	/*
		Parallel multiply given input matrices using optimal methods and return resultant matrix
	*/

	int i, j, k, iOff, jOff;
	double elapsed;
	TYPE tot;
	struct timeval t0, t1;
	
	gettimeofday(&t0, 0);

	/* Head */
	convert(matrixA, matrixB, dimension);
	#pragma omp parallel shared(matrixC) private(i, j, k, iOff, jOff, tot) num_threads(4)
	{
		#pragma omp for schedule(static)
		for(i=0; i<dimension; i++){
			iOff = i * dimension;
			for(j=0; j<dimension; j++){
				jOff = j * dimension;
				tot = 0;
				for(k=0; k<dimension; k++){
					tot += flatA[iOff + k] * flatB[jOff + k];
				}
				matrixC[i][j] = tot;
			}
		}
	}
	/* Tail */

	gettimeofday(&t1, 0);
	elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	
	return elapsed;
}

void convert(TYPE** matrixA, TYPE** matrixB, int dimension){
	#pragma omp parallel for
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			flatA[i * dimension + j] = matrixA[i][j];
			flatB[j * dimension + i] = matrixB[i][j];
		}
	}
}

void dominateset(double ***sparse_mat,double ***mat,int knn,int n,int row,int col)
{
	int i,j,k,p,*minindx,*flag;
	double *knearest;
	double sum;
	char fname[100];
	FILE *fp;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<row;j++)
		{
			flag=(int*)calloc(col,sizeof(int));
			knearest=(double *)malloc(knn*sizeof(double));
			minindx=(int *)malloc(knn*sizeof(int));
			
			p=0;
			while(p<knn)
			{
				for(k=0;k<col;k++)
				{
					if(flag[k]==0)
					{
						knearest[p]=mat[i][j][k]; 
						minindx[p]=k;
						break;
					}
				}
				
				for(k=0;k<col;k++)
				{
					if((mat[i][j][k]>knearest[p])&&(flag[k]==0))
					{
						knearest[p]=mat[i][j][k];
						minindx[p]=k;
					}
				}
				flag[minindx[p]]=1;
				sparse_mat[i][j][minindx[p]]=knearest[p];
				p++;
			}
			free(flag);
			free(knearest);
			free(minindx);
		}
		
		for(j=0;j<row;j++)
		{
			sum=0.0;
			for(k=0;k<col;k++)
				sum+=sparse_mat[i][j][k];
			if(sum!=0.0)
			{
				for(k=0;k<col;k++)
					sparse_mat[i][j][k]=sparse_mat[i][j][k]/(double)sum;
			}
			else
			{
				for(k=0;k<col;k++)
					sparse_mat[i][j][k]=0.0;
				/*printf("Error Encountered...All zero-row found\n");
				printf("i=%d\tj=%d\tk=%d\n",i,j,k);
				fflush(stdout);
				exit(0);*/
			}
		}
	}
	
	return;
}

double ***scale3D(double ***mat,int n, int row,int col)
{
	int i,j,k;
	double min,max;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<row;j++)
		{
			max=mat[i][j][0];
			min=mat[i][j][0];
			for(k=0;k<col;k++)
			{
				if(mat[i][j][k]>max)
					max=mat[i][j][k];
				if(mat[i][j][k]<min)
					min=mat[i][j][k];
			}
			
			for(k=0;k<col;k++)
				mat[i][j][k]=(mat[i][j][k]-min)/(double)(max-min);
		}
	}
	
	return mat;
}
	
		

void normalize3D(double ***mat,int n,int row,int col)
{
	int i,j,k;
	double sum;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<row;j++)
		{
			sum=0.0;
			for(k=0;k<col;k++)
				if(j!=k)
					sum+=mat[i][j][k];
			
			if(sum!=0.0)
			{
				for(k=0;k<col;k++)
					if(j==k)
						mat[i][j][k]=0.5;
					else
						mat[i][j][k]=mat[i][j][k]/(double)(2*sum);
			}
			else
			{
				for(k=0;k<col;k++)
					mat[i][j][k]=0.0;
				/*printf("Error Encountered...All zero-row found\n");
				printf("i=%d\tj=%d\tk=%d\n",i,j,k);
				fflush(stdout);
				exit(0);*/
			}
		}
	}
	
	return;
}

void normalize2D(double **mat,int row,int col)
{
	int j,k;
	double sum;
	
	for(j=0;j<row;j++)
	{
		sum=0.0;
		for(k=0;k<col;k++)
			if(j!=k)
				sum+=mat[j][k];
			
		if(sum!=0.0)
		{
			for(k=0;k<col;k++)
				if(j==k)
					mat[j][k]=0.5;
				else
					mat[j][k]=mat[j][k]/(double)(2*sum);
		}
		else
		{
			for(k=0;k<col;k++)
				mat[j][k]=0.0;
			/*printf("Error Encountered...All zero-row found\n");
			printf("j=%d\tk=%d\n",j,k);
			fflush(stdout);
			exit(0);*/
		}
	}
	return;
}

void symmetricity(double ***mat,int n,int row,int col)
{
	int i,j,k;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<row;j++)
		{
			for(k=j;k<col;k++)
			{
				mat[i][j][k]=(mat[i][j][k]+mat[i][k][j])/(double)2;
				mat[i][k][j]=mat[i][j][k];
			}
		}
	}
	
	return;
}
	

double ***allocate3D(int n,int row,int col)
{
	int i,j,k;
	double ***mat;
	
	mat=(double ***)malloc(n*sizeof(double **));
	if(mat!=NULL)
	{
		for(i=0;i<n;i++)
		{
			mat[i]=(double **)malloc(row*sizeof(double *));
			if(mat[i]!=NULL)
			{
				for(j=0;j<row;j++)
				{
					mat[i][j]=(double *)calloc(col,sizeof(double));
					if(mat[i][j]==NULL)
					{
						printf("Insufficient Memory\n");
						exit(0);
					}
				}
			}
		}
	}
	
	return mat;
}

double **allocate2D(int row,int col)
{
	int i;
	double **mat;
	
	mat=(double **)malloc(row*sizeof(double *));
	for(i=0;i<row;i++)
		mat[i]=(double *)calloc(col,sizeof(double));
		
	return mat;
}

void deallocate3D(double ***mat,int n,int row)
{
	int i,j;
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<row;j++)
			free(mat[i][j]);
		free(mat[i]);
	}
	free(mat);
	
	return;
}

void deallocate2D(double **mat, int row)
{
	int i;
	
	for(i=0;i<row;i++)
		free(mat[i]);
	free(mat);
	
	return;
}
