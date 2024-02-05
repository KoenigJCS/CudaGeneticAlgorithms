#include <iostream>

#define MAXPOP 32768
#define MAX_CHROMLENGTH 100
#define MAX_VERTICES 100

#include <math.h>

#include <fstream>
#include <iostream>

#include <string>
#include <assert.h>
#include <cstring>
#include <random>
#include <chrono>

using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

extern std::mt19937 MyRandom;

void InitializeRandomNumberGenerator(int seed);

void WriteBufToFile(std::string buf, std::string filename);

float RandomFraction();
int Flip(float prob);
int IntInRange(int low, int high);

std::mt19937 MyRandom;
std::uniform_real_distribution<float> uniformFloat01Distribution;

void WriteBufToFile(std::string buf, std::string filename){
	std::ofstream ofs(filename, std::ofstream::app);
	if(ofs.good()){
		ofs << buf;
	}
	ofs.flush();
	ofs.close();
}

void InitializeRandomNumberGenerator(int seed) {
	MyRandom = std::mt19937(seed);
	uniformFloat01Distribution = std::uniform_real_distribution<float> (0.0f, 1.0f);
}


int Flip(float prob){
	int f = (MyRandom() < prob * MyRandom.max() ? 1 : 0);
	return f;
}

/* greater than equal to low and strictly less than high */
int IntInRange(int low, int high){
	return low + MyRandom()%(high - low);
}

/* greater than equal to 0 and less than 1 */
float RandomFraction(){
	return ((float)(MyRandom() % 1000)) / (float)1000.0;
	//return uniformFloat01Distribution(random);
}

typedef struct {
	std::string infile;
	std::string outfile;

	//std::string graphInfile;

	long int randomSeed;
	int popSize;
	int chromLength;
	unsigned int maxgens;
	float px;
	float pm;

} Options;

struct Individual {
public:
	Individual(int chromLen);
	Individual();

	int chromLength;
	void Init();
	void Mutate(float pm);
	double fitness;
	int chromosome[MAX_CHROMLENGTH];
};

class Population {
public:
	Population(Options options);
	virtual ~Population();
	//------------------------

	Options options;
	Individual members[MAXPOP];
	Individual *members_d;
	double avg, min, max, sumFitness;


	void Init();
	void Evaluate();
	void EvaluateLower();
	void Generation(Population *child);
	void Report(unsigned long int gen);
	void Statistics();

	int ProportionalSelector();
	void XoverAndMutate(Individual *p1, Individual *p2, Individual *c1, Individual *c2);
	void TwoPoint(Individual *p1, Individual *p2, Individual *c1, Individual *c2);
	void OnePoint(Individual *p1, Individual *p2, Individual *c1, Individual *c2);

	void halve(Population *child);
	void CHCGeneration(Population *child);

	std::string ToString(int start, int end);
};


double Eval(Individual *individual){
	double sum = 0;
	//_sleep(10);
	printf("REEEEEEEEEEEEE!");
	for(int i = 0; i < individual->chromLength; i++){
		sum += (individual->chromosome[i] == 1 ? 1: 0);
	}
	return sum;
}

double Eval(Individual *individual);

class GA {

private:
	//void Construct();

public:
	GA(int argc, char *argv[]);

	virtual ~GA();

	//--------------------------------
	Population *parent;
	Population *child;
	Options options;


	void SetupOptions(int argc, char*argv[]);

	void Init();
	void Run();
	void CHCRun();

};

int main(int argc, char * argv[])
{
	std::cout << "CHC genetic algorithm: " << argv[0] << std::endl; 
	//std::cout << "MyRandom:"<<MyRandom()<<","<<MyRandom()<<","<<MyRandom()<<","<<MyRandom()<<","<<MyRandom()<<","<<std::endl;
	GA ga(argc, argv);

	ga.Init();
	auto beg = std::chrono::high_resolution_clock::now();
	ga.CHCRun();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
 
    // Displaying the elapsed time
    
	if(duration.count()<1000.0f)
		std::cout << "Elapsed Time(ms): " << duration.count();
	else
		std::cout << "Elapsed Time(s): " << duration.count()/1000.0f;
	return 0;
}
Individual::Individual()
{
	chromLength = MAX_CHROMLENGTH;
	fitness = -1;
}

Individual::Individual(int len) {
	// TODO Auto-generated constructor stub  
	chromLength = len;
	fitness = -1;
}


void Individual::Init(){
	for(int i = 0; i < chromLength; i++){
		chromosome[i] = Flip(0.5f);
	}
}

void Individual::Mutate(float pm){
	for(int i = 0; i < chromLength; i++){
		if(Flip(pm)){
			chromosome[i] = 1 - chromosome[i];
			//std::cout<<"Mutate!\n";

		}
	}

}


Population::Population(Options opts) {
	options = opts;	
	avg = min = max = sumFitness = -1;
	cudaMalloc((void **)&members_d,options.popSize*2*sizeof(Individual));
	assert(options.popSize * 2 <= MAXPOP);
	for (int i = 0; i < options.popSize * 2; i++){
		members[i] = struct Individual(options.chromLength);
		members[i].Init();
	}
}

Population::~Population() {
	cudaFree(members_d);
	// TODO Auto-generated destructor stub
}



void Population::Init(){
	
	Evaluate();
}

__global__ void EvalGPU(Individual* members, int length)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index<length)
    {
		//printf("Here at:%i\n",index);
		double sum = 0;
		// for(int i = 0; i < members[index].chromLength; i++){
		// 	sum += (members[index].chromosome[i] == 1 ? 1: 0);
		// }
		for(int o = 0; o < 3000; o++)
		{
			for(int i = 0; i < members[index].chromLength; i++)
			{
				if(i%2==1)
					sum+=members[index].chromosome[i];
			}
			for(int i = 0; i < members[index].chromLength; i++)
			{
				if(i%2==0)
					sum-=members[index].chromosome[i];
			}
		}
		//printf("%lf",sum);
		members[index].fitness=sum;
	}
}


void Population::Evaluate(){
	size_t size = options.popSize*2*sizeof(Individual);

	//printf("%i,%i,%i,%i",(int)sizeof(int),(int)sizeof(int)*options.chromLength,(int)sizeof(Individual),size);

	
	cudaMemcpy(members_d,members,size,cudaMemcpyHostToDevice);
	EvalGPU<<<(options.popSize*2+255)/256, 256>>>(members_d,options.popSize*2);
	cudaMemcpy(members,members_d,size,cudaMemcpyDeviceToHost);

	//free(fitValues);
	//cudaFree(fitValues_d);
	// for (int i = 0; i < options.popSize; i++){
	// 	members[i].fitness = Eval(&members[i]);
	// }
}

void Population::EvaluateLower(){
	size_t size = options.popSize*sizeof(Individual);

	//printf("%i,%i,%i,%i",(int)sizeof(int),(int)sizeof(int)*options.chromLength,(int)sizeof(Individual),size);

	
	cudaMemcpy(members_d+size,members+size,size,cudaMemcpyHostToDevice);
	EvalGPU<<<(options.popSize+255)/256, 256>>>(members_d+size,options.popSize);
	cudaMemcpy(members+size,members_d+size,size,cudaMemcpyDeviceToHost);
	//free(fitValues);
	//cudaFree(fitValues_d);
	// for (int i = 0; i < options.popSize; i++){
	// 	members[i].fitness = Eval(&members[i]);
	// }
}

void Population::Statistics(){
	sumFitness = 0;
	min = members[0].fitness;
	max = members[0].fitness;
	for(int i = 0; i < options.popSize; i++){
		sumFitness += members[i].fitness;
		if(min > members[i].fitness)
			min = members[i].fitness;
		if(max < members[i].fitness)
			max = members[i].fitness;
	}
	avg = sumFitness/options.popSize;
}

void Population::Report(unsigned long int gen){
	char printbuf[1024];
	sprintf(printbuf, "%4i \t %f \t %f \t %f\n ", (int)gen, min, avg, max);
	WriteBufToFile(std::string(printbuf), options.outfile);
	std::cout << printbuf;
}

void Population::Generation(Population *child){
	int pi1, pi2, ci1, ci2;
	Individual *p1, *p2, *c1, *c2;
	for (int i = 0; i < options.popSize; i += 2) {
		pi1 = ProportionalSelector();
		pi2 = ProportionalSelector();

		ci1 = i;
		ci2 = i + 1;

		p1 = &(members[pi1]); p2 = &(members[pi2]);
		c1 = &(child->members[ci1]); c2 = &(child->members[ci2]);

		XoverAndMutate(p1, p2, c1, c2);
	}
}

void Population::CHCGeneration(Population *child) {
	int pi1, pi2, ci1, ci2;
	Individual *p1, *p2, *c1, *c2;
	
	//std::cout << "Parents" << std::endl;
	//std::cout << ToString(0, options.popSize);
	
	for (int i = 0; i < options.popSize; i += 2) {
		pi1 = ProportionalSelector();
		pi2 = ProportionalSelector();

		ci1 = options.popSize + i;
		ci2 = options.popSize + i + 1;

		p1 = &(members[pi1]); p2 = &(members[pi2]);
		c1 = &(members[ci1]); c2 = &(members[ci2]);

		XoverAndMutate(p1, p2, c1, c2);
	}
	halve(child);
}

int compareFitness(const void *x, const void *y) {
  Individual *a = (Individual *) x;
  Individual *b = (Individual *) y;
//   std::cout << "before comparison\n";
//   std::cout << "comparing " << a->fitness << " against " << b->fitness << std::endl;
  return (int) (a->fitness - b->fitness);
}

void Population::halve(Population*child) {
	//child->Evaluate();
//   for (int i = options.popSize; i < 2 * options.popSize; i++){
//     members[i].fitness = Eval(&members[i]);
//   }
  //std::cout << "Intermediate\n";
  //std::cout << ToString(options.popSize, options.popSize * 2);
  qsort(members, options.popSize * 2, sizeof(Individual), compareFitness);
  //std::cout << "Sorted Parents + children\n";
  //std::cout << ToString(0, options.popSize * 2);
  for(int i = 0; i < options.popSize; i++){
	  memcpy(&(child->members[i]), &(members[options.popSize + i]), sizeof(Individual));
  }
  //std::cout << "Children\n";
  //std::cout << child->ToString(0, options.popSize);  
}

std::string Population::ToString(int start, int end){
  std::string s = "";
  for(int i = start; i < end; i++){
    s = s + "------------\n";
  }
  return s;
}

int Population::ProportionalSelector(){
	int i = -1;
	double sum = 0;
	double limit = RandomFraction() * sumFitness;
	do {
		i = i + 1;
		sum += members[i].fitness;
	} while (sum < limit && i < options.popSize-1 );
	//std::cout << i << " \n";
	return i;
}

void Population::XoverAndMutate(Individual *p1, Individual *p2, Individual *c1, Individual *c2){

	for(int i = 0; i < options.chromLength; i++){ //First copy
		c1->chromosome[i] = p1->chromosome[i];
		c2->chromosome[i] = p2->chromosome[i];
	}
	if(Flip(options.px)){ // if prob, then cross/exchange bits
		TwoPoint(p1, p2, c1, c2);
	}

	c1->Mutate(options.pm);
	c2->Mutate(options.pm);
}

void Population::OnePoint(Individual *p1, Individual *p2, Individual *c1, Individual *c2){ //not debugged
	int t1 = IntInRange(0, options.chromLength);
	for(int i = t1; i < options.chromLength; i++){
		c1->chromosome[i] = p2->chromosome[i];
		c2->chromosome[i] = p1->chromosome[i];
	}
}

void Population::TwoPoint(Individual *p1, Individual *p2, Individual *c1, Individual *c2){ //not debugged
	int t1 = IntInRange(0, options.chromLength);
	int t2 = IntInRange(0, options.chromLength);
	int xp1 = std::min(t1, t2);
	int xp2 = std::max(t1, t2);
	//std::cout<<"Swapping:"<<xp1<<" and "<<xp2<<'\n';
	for(int i = xp1; i < xp2; i++){
		c1->chromosome[i] = p2->chromosome[i];
		c2->chromosome[i] = p1->chromosome[i];
	}
}

GA::GA(int argc, char *argv[]) {
	SetupOptions(argc, argv);
	InitializeRandomNumberGenerator(options.randomSeed);
}

GA::~GA() {
	// TODO Auto-generated destructor stub
}


void GA::SetupOptions(int argc, char *argv[]){
	options.randomSeed = time(NULL);
	//Make sure to make pop size a power of two if you can help it 
	options.popSize = 4096;
	options.chromLength = 10;
	options.maxgens = 5;
	options.px = 0.95f;
	options.pm = 0.1f;
	options.infile = std::string ("infile");
	options.outfile = std::string("outfile");
	//options.graphInfile = std::string("graph-raw.csv");
}

void GA::Init(){
	//EvalInit(options);
	parent = new Population(options);
	child  = new Population(options);
	parent->Init(); // evaluates, stats, and reports on initial population
	parent->Statistics();
	parent->Report(0);

}

void GA::Run(){
	for (unsigned long int i = 1; i < options.maxgens; i++) {
		parent->Generation(child);
		child->Evaluate();
		child->Statistics();
		child->Report(i);

		Population *tmp = parent;
		parent = child;
		child = tmp;
	}

}

void GA::CHCRun() {
	for (unsigned long int i = 1; i < options.maxgens; i++) {
		parent->CHCGeneration(child);
		child->Statistics();
		child->Report(i);

		Population *tmp = parent;
		parent = child;
		child = tmp;
	}
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
