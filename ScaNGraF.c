#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/stat.h>

void get_zpowers(int ,int ,int ,char *,int knn);
void compute_z(char *,char *,char *,char *);
void createRProductscript(char *,char *,char *,char *);
void dominateset(double ***,int ,int ,int ,int ,char *);
void normalize3D(double ***,int ,int ,int );
void normalize2D(double **,int,int);
double ***scale3D(double ***,int , int ,int );
void symmetricity(double ***,int,int,int);
double ***allocate3D(int,int,int);
double **allocate2D(int,int);
void deallocate2D(double **,int);
void deallocate3D(double ***,int,int);

int main(int argc,char *argv[])
{
	int n,i,j,k,t;
	int p,q;
	int row, col;
	int flag,knn;
	int start1,end;
	int diff;
	double **temp;
	double ***mat;
	double ***new_mat;
	char file[1000];
	char file1[1000];
	char file2[1000];
	char file3[1000];
	char comand[1000];
	char product[1000];
    struct stat st;
    time_t start, stop;
	FILE *fp;
	
	if(argc==1)
	{
		printf("Enter the following information:\n");
		printf("1. Number of Similarity Files... (Note: Name the network files as NETWORK1.txt, NETWORK2.txt,...)\n");
		printf("2. Number of Iterations\n");
		printf("3. Number of nearest neighbors\n");
		printf("4. Path to Similarity Matrices\n");
		exit(0);
	}
	
	time(&start);
	n=atoi(argv[1]);
	
	knn=atoi(argv[3]);
	
	for(i=0;i<n;i++)
	{
		sprintf(comand,"wc -l %s/NETWORK%d.txt > temp.txt",argv[4],i+1);
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
		sprintf(file,"%s/NETWORK%d.txt",argv[4],i+1);
		fp=fopen(file,"r");
		for(j=0;j<row;j++)
			for(k=0;k<col;k++)
				fscanf(fp,"%lf\t",&mat[i][j][k]);
		fclose(fp);
	}
	mat=scale3D(mat,n,row,col);
		
	flag=0;
	for(i=0;i<n;i++)
	{
		for(j=0;j<row;j++)
		{
			if(mat[i][j][j]!=0.5)
			{
				flag=1;
				break;
			}
		}
	}
	
	if(flag==1)
	{
		normalize3D(mat,n,row,col);
		symmetricity(mat,n,row,col);
		printf("Networks are now Symmetric and Normalized\n");
	
		for(i=0;i<n;i++)
		{
			sprintf(file,"%s/Global%d.txt",argv[4],i+1);
			fp=fopen(file,"w");
			for(j=0;j<row;j++)
			{
				for(k=0;k<col-1;k++)
					fprintf(fp,"%.17lf\t",mat[i][j][k]);
				fprintf(fp,"%.17lf\n",mat[i][j][k]);
			}
			fclose(fp);
		}
		printf("Updated Global Matrices written to file\n");
	}
	
	printf("Considering kernel with %d neighbors\n",knn);
	for(i=0;i<n;i++)
	{
		sprintf(file,"%s/Sparse%d_k%d.txt",argv[4],i+1,knn);
		if(stat(file,&st)==0)
			printf("Local Affinity Matrix Found\n");
		else
		{
			dominateset(mat,knn,n,row,col,argv[4]);
			printf("Computing Local Affinity Matrix\n");
		}
	}
	deallocate3D(mat,n,row);
	
	/*start1=atoi(argv[2]);
	if(start1<0)
	{
		printf("Initial Value of Start is 0\n");
		fflush(stdout);
		exit(0);
	}
	end=atoi(argv[3]);
	t=(end-start1);*/
	t=atoi(argv[2]);
	printf("No. of iterations: %d\n",t);
	
	if(t==0)
	{
		printf("Zero iterations selected\n");
		printf("Program exiting\n");
		exit(0);
	}
	
	if(t==1)
	{
		new_mat=allocate3D(n,row,col);
		for(i=0;i<n;i++)
		{
			k=0;
			mat=allocate3D(n-1,row,col);
			for(j=0;j<n;j++)
			{
				if(i!=j)
				{
					sprintf(file1,"%s/Global%d.txt",argv[4],j+1);
					sprintf(file2,"%s/Sparse%d_k%d.txt",argv[4],i+1,knn);
					sprintf(file3,"%s/Global%d_%d_t2.txt",argv[4],i+1,j+1);
					if(stat(file3,&st)!=0)
					{
						strcpy(product,"product2.R");
						createRProductscript(file1,file2,file3,product);
						system("R CMD BATCH product2.R");
						system("cat product2.Rout");
					}
		
					fp=fopen(file3,"r");
					for(p=0;p<row;p++)
						for(q=0;q<col;q++)
							fscanf(fp,"%lf",&mat[k][p][q]);
					fclose(fp);
					k++;
				}
			}
			
			for(p=0;p<row;p++)
				for(q=0;q<col;q++)
				{
					for(j=0;j<n-1;j++)
						new_mat[i][p][q]+=mat[j][p][q];
					new_mat[i][p][q]=new_mat[i][p][q]/(double)(n-1);
				}
		}
	}
	
	if(t>1)
	{
		new_mat=allocate3D(n,row,col);
		for(i=0;i<n;i++)
		{
			k=0;
			mat=allocate3D(n-1,row,col);
			for(j=0;j<n;j++)
			{
				if(j!=i)
				{
					//Computing product of ith and jth sparse matrix
					sprintf(file1,"%s/Sparse%d_k%d.txt",argv[4],i+1,knn);
					sprintf(file2,"%s/Sparse%d_k%d.txt",argv[4],j+1,knn);
					sprintf(file3,"%s/Sparse%d_%d_k%d_t1.txt",argv[4],i+1,j+1,knn);
					if(stat(file3,&st)!=0)
					{
						strcpy(product,"compute_z1.R");
						compute_z(file1,file2,file3,product);
						system("R CMD BATCH compute_z1.R");
						system("cat compute_z1.Rout");
					}
					printf("Product of Sparse%d and Sparse%d Computed\n",i+1,j+1);
				
					if(t%2==0)
					{
						if((t/2)>1)
							get_zpowers(i+1,j+1,t/2,argv[4],knn);
						printf("Powers of Sparse%d_%d computed\n",i+1,j+1);
					
						sprintf(file1,"%s/Global%d.txt",argv[4],i+1);
						sprintf(file2,"%s/Sparse%d_%d_k%d_t%d.txt",argv[4],i+1,j+1,knn,t/2);
						sprintf(file3,"%s/Global%d_%d_t%d.txt",argv[4],i+1,j+1,t+1);
						if(stat(file3,&st)!=0)
						{
							strcpy(product,"product2.R");
							createRProductscript(file1,file2,file3,product);
							system("R CMD BATCH product2.R");
							system("cat product2.Rout");
							printf("Global%d_%d_t%d.txt Created\n",i+1,j+1,t);
						}
					}
					else
					{
						if(((t-1)/2)>1)
							get_zpowers(i+1,j+1,(t-1)/2,argv[4],knn);
						printf("Powers of Sparse%d_%d computed\n",i+1,j+1);
					
						sprintf(file1,"%s/Sparse%d_%d_k%d_t%d.txt",argv[4],i+1,j+1,knn,(t-1)/2);
						sprintf(file2,"%s/Sparse%d_k%d.txt",argv[4],i+1,knn);
						sprintf(file3,"%s/Sparse%d_%d_t%d_Sparse%d.txt",argv[4],i+1,j+1,(t-1)/2,i+1);
						if(stat(file3,&st)!=0)
						{
							strcpy(product,"compute_z1.R");
							compute_z(file1,file2,file3,product);
							system("R CMD BATCH compute_z1.R");
							system("cat compute_z1.Rout");
						}
					
						sprintf(file1,"%s/Global%d.txt",argv[4],j+1);
						sprintf(file2,"%s/Sparse%d_%d_t%d_Sparse%d.txt",argv[4],i+1,j+1,(t-1)/2,i+1);
						sprintf(file3,"%s/Global%d_%d_t%d.txt",argv[4],i+1,j+1,t+1);
						if(stat(file3,&st)!=0)
						{
							strcpy(product,"product2.R");
							createRProductscript(file1,file2,file3,product);
							system("R CMD BATCH product2.R");
							system("cat product2.Rout");
						}
						printf("Global%d_%d_t%d.txt Created\n",i+1,j+1,t);
					}
					fp=fopen(file3,"r");
					for(p=0;p<row;p++)
						for(q=0;q<col;q++)
							fscanf(fp,"%lf",&mat[k][p][q]);
					fclose(fp);
					k++;
				}
			}
		
			if(n>2)
			{
				for(p=0;p<row;p++)
				{
					for(q=0;q<col;q++)
					{
						for(j=0;j<n-1;j++)
							new_mat[i][p][q]+=mat[j][p][q];
						new_mat[i][p][q]=new_mat[i][p][q]/(double)(n-1);
					}
				}
			}
			else
			{
				for(p=0;p<row;p++)
					for(q=0;q<col;q++)
						new_mat[i][p][q]=mat[0][p][q];
			}
		
			deallocate3D(mat,n-1,row);
		
			sprintf(file1,"%s/Global%d_t%d.txt",argv[4],i+1,t+1);
			fp=fopen(file1,"w");
			for(p=0;p<row;p++)
			{
				for(q=0;q<col-1;q++)
					fprintf(fp,"%.17lf\t",new_mat[i][p][q]);
				fprintf(fp,"%.17lf\n",new_mat[i][p][q]);
			}
		}
	}
	
	temp=allocate2D(row,col);
	for(j=0;j<row;j++)
	{
		for(k=0;k<col;k++)
		{
			for(i=0;i<n;i++)
				temp[j][k]+=new_mat[i][j][k];
			temp[j][k]=temp[j][k]/(double)n;
		}
	}
	
	normalize2D(temp,row,col);
	for(j=0;j<row;j++)
	{
		for(k=j;k<col;k++)
		{
			temp[j][k]=(temp[j][k]+temp[k][j])/(double)2;
			temp[k][j]=temp[j][k];
		}
	}
	
	sprintf(file1,"%s/FUSION_K_%d_ITR_%d.txt",argv[4],knn,end);
	fp=fopen(file1,"w");
	for(i=0;i<row;i++)
	{
		for(j=0;j<col-1;j++)
			fprintf(fp,"%.17lf\t",temp[i][j]);
		fprintf(fp,"%.17lf\n",temp[i][j]);
	}
	fclose(fp);
	
	time(&stop);
	sprintf(file1,"%s/SEQUENTIAL_EXECUTION_TIME.txt",argv[4]);
	fp=fopen(file,"w");
	fprintf(fp,"K=%d\tT=%d\tTime=%.2lf secs\n",knn,end,difftime(stop,start));
	fclose(fp);
	
	printf("Time Taken: %lf secs\n",difftime(stop,start));
	printf("Diffusion Completed\n");
	
	deallocate2D(temp,row);
	deallocate3D(new_mat,n,row);
	
	return 0;
}

void get_zpowers(int li,int lj,int t,char *path,int knn)
{
	char file1[1000];
	char file2[1000];
	char file3[1000];
	char file[1000];
	char product[1000];
	int iter;
	int diff;
	int i;
	struct stat st;
	
	iter=2;
	while(iter<=t)
	{
		sprintf(file1,"%s/Sparse%d_%d_k%d_t%d.txt",path,li,lj,knn,iter/2);
		sprintf(file2,"%s/Sparse%d_%d_k%d_t%d.txt",path,li,lj,knn,iter);
		if(stat(file2,&st)!=0)
		{
			strcpy(product,"compute_z1.R");
			compute_z(file1,file1,file2,product);
			system("R CMD BATCH compute_z1.R");
			system("cat compute_z1.Rout");
			printf("(Sparse%d_%d)^%d computed\n",li,lj,iter);
		}
		iter=2*iter;
	}
	iter=iter/2;
	
	diff=t-iter;
	printf("Diff:%d\n",diff);
	if(diff>0)
	{
		sprintf(file1,"%s/Sparse%d_%d_k%d_t%d.txt",path,li,lj,knn,iter);
		sprintf(file2,"%s/Sparse%d_%d_k%d_t%d.txt",path,li,lj,knn,diff);
		if(stat(file2,&st)!=0)
			get_zpowers(li,lj,diff,path,knn);
		sprintf(file3,"%s/Sparse%d_%d_k%d_t%d.txt",path,li,lj,knn,iter+diff);
		if(stat(file3,&st)!=0)
		{
			strcpy(product,"compute_z1.R");
			compute_z(file1,file2,file3,product);
			system("R CMD BATCH compute_z1.R");
			system("cat compute_z1.Rout");
		}
		printf("Sparse%d_%d_k%d_t%d.txt computed\n",li,lj,knn,iter+diff);
	}
		
	return;
}

void createRProductscript(char *ipfile1,char *ipfile2,char *opfile,char *execR)
{
	char file1[1000],file2[1000];
	FILE *fp;
	
	fp=fopen(execR,"w");
	fprintf(fp,"library(SparseM)\n");
	fprintf(fp,"library(data.table)\n");
	fprintf(fp,"product<-function(ipfile1,ipfile2,opfile){\n");
	fprintf(fp,"cwd<-getwd()\n");
	fprintf(fp,"fname1<-paste(cwd,ipfile1,sep=\"/\")\n");
	fprintf(fp,"fname2<-paste(cwd,ipfile2,sep=\"/\")\n");
	fprintf(fp,"fname3<-paste(cwd,opfile,sep=\"/\")\n");
	fprintf(fp,"print(fname1)\n");
	fprintf(fp,"print(fname2)\n");
	fprintf(fp,"print(fname3)\n");
	fprintf(fp,"mat<-as.matrix(fread(file=fname1,sep=\"\t\",header=FALSE))\n");
	fprintf(fp,"spmat<-as.matrix.csr(as.matrix(fread(file=fname2,sep=\"\t\",header=FALSE)))\n");
	fprintf(fp,"prod<-as.matrix(spmat %s mat %s t(spmat))\n","%*%","%*%");
	fprintf(fp,"write.table(prod,file=opfile,sep=\"\t\",row.names=FALSE,col.names=FALSE)\n");
	fprintf(fp,"remove(list=ls())\n}\n\n");
	fprintf(fp,"product(\"%s\",\"%s\",\"%s\")\n",ipfile1,ipfile2,opfile);
	fclose(fp);
	
	return;
}
	

void compute_z(char *ipfile1,char *ipfile2,char *opfile,char *execR)
{
	char file1[1000],file2[1000];
	FILE *fp;
	
	fp=fopen(execR,"w");
	fprintf(fp,"library(data.table)\n");
	fprintf(fp,"get_z<-function(ipfile1,ipfile2,opfile){\n");
	fprintf(fp,"cwd<-getwd()\n");
	fprintf(fp,"fname1<-paste(cwd,ipfile1,sep=\"/\")\n");
	fprintf(fp,"fname2<-paste(cwd,ipfile2,sep=\"/\")\n");
	fprintf(fp,"fname3<-paste(cwd,opfile,sep=\"/\")\n");
	fprintf(fp,"print(fname1)\n");
	fprintf(fp,"print(fname2)\n");
	fprintf(fp,"print(fname3)\n");
	fprintf(fp,"spmat1<-as.matrix(fread(file=fname1,sep=\"\t\",header=FALSE))\n");
	fprintf(fp,"spmat2<-as.matrix(fread(file=fname2,sep=\"\t\",header=FALSE))\n");
	fprintf(fp,"prod<-as.matrix(spmat1 %s spmat2)\n","%*%");
	fprintf(fp,"write.table(prod,file=opfile,sep=\"\t\",row.names=FALSE,col.names=FALSE)\n");
	fprintf(fp,"remove(list=ls())\n}\n\n");
	fprintf(fp,"get_z(\"%s\",\"%s\",\"%s\")\n",ipfile1,ipfile2,opfile);
	fclose(fp);
	
	return;
}
	

void dominateset(double ***mat,int knn,int n,int row,int col,char *path)
{
	int i,j,k,p,*minindx,*flag;
	double ***sparse_mat,*knearest;
	double sum;
	char fname[100];
	FILE *fp;
	
	sparse_mat=allocate3D(n,row,col);
	
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
		
		sprintf(fname,"%s/Sparse%d_k%d.txt",path,i+1,knn);
		fp=fopen(fname,"w");
		for(j=0;j<row;j++)
		{
			for(k=0;k<col-1;k++)
				fprintf(fp,"%.17lf\t",sparse_mat[i][j][k]);
			fprintf(fp,"%.17lf\n",sparse_mat[i][j][k]);
		}
		fflush(fp);
		fclose(fp);
	}
	
	deallocate3D(sparse_mat,n,row);
	
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
