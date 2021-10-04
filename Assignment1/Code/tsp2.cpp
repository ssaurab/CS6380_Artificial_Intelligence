#include <bits/stdc++.h>
#include "tsp2.h"
using namespace std;
//do we need sanity checkss

Graph::Graph(int V, int initial_vertex, bool random_graph) 
{
	this->V = V; 
	this->initial_vertex = initial_vertex; 
	this->total_edges = 0; 
}

void Graph::addEdge(int src, int dest, float weight) 
{
	map_edges[make_pair(src, dest)] = weight; 
}

float Graph::edgeCost(int src, int dest) 
{
	map<pair<int, int>,float>::iterator it = map_edges.find(make_pair(src, dest));
	return it->second; 
}

Genetic::Genetic(Graph* graph, int size_population, int generations, int mutation_rate, bool show_population)
{
	this->graph = graph;
	this->size_population = size_population;
	this->real_size_population = 0;
	this->generations = generations;
	this->mutation_rate = mutation_rate;
	this->show_population = show_population;
}

float Genetic::isValidSolution(vector<int>& solution)
{
	float total_cost = 0;
	set<int> set_solution;
	
	for(int i = 0; i < graph->V; i++)
		set_solution.insert(solution[i]);
	
	if(set_solution.size() != (unsigned)graph->V)
		return -1;

	for(int i = 0; i < graph->V; i++)
	{
		if(i + 1 <  graph->V)
		{
			float cost = graph->edgeCost(solution[i], solution[i+1]);
			if(cost == -1)
				return -1;
			else
				total_cost += cost;
		}
		else
		{
			float cost = graph->edgeCost(solution[i], solution[0]);
			if(cost == -1)
				return -1;
			else
				total_cost += cost;
			break;
		}
	}
	return total_cost;
}

bool Genetic::existsChromosome(const vector<int> & v)
{
	for(vector<pair<vector<int>, float> >::iterator it=population.begin(); it!=population.end(); ++it)
	{
		const vector<int>& vec = (*it).first; 
		if(equal(v.begin(), v.end(), vec.begin())) 
			return true;
	}
	return false;
}

void Genetic::initialPopulation() 
{
	vector<int> parent;
	parent.push_back(graph->initial_vertex);
	for(int i = 0; i < graph->V; i++)
	{
		if(i != graph->initial_vertex)
			parent.push_back(i);
	}	
	float total_cost = isValidSolution(parent);

	population.push_back(make_pair(parent, total_cost)); 
	real_size_population++; 
	
	for(int i = 0; i < 10000 && real_size_population != size_population; i++)
	{
		random_shuffle(parent.begin() + 1, parent.begin() + (rand() % (graph->V - 1) + 1));
		
		float total_cost = isValidSolution(parent); 
		
		if(!existsChromosome(parent))
		{
			population.push_back(make_pair(parent, total_cost)); 
			real_size_population++; 
		}
	}
	sort(population.begin(), population.end(), sort_pred()); 
}

void Genetic::showPopulation()
{
	cout << "\nShowing solutions...\n\n";
	for(vector<pair<vector<int>, float> >::iterator it=population.begin(); it!=population.end(); ++it)
	{
		const vector<int>& vec = (*it).first; 
		
		for(int i = 0; i < graph->V; i++)
			cout << vec[i] << " ";
		cout << " | Cost: " << (*it).second << "\n\n";
	}
	cout << "\nPopulation size: " << real_size_population << endl;
}

void Genetic::insertBinarySearch(vector<int>& child, float total_cost)
{
	int imin = 0;
	int imax = real_size_population - 1;
	
	while(imax >= imin)
	{
		int imid = imin + (imax - imin) / 2;
		if(total_cost == population[imid].second)
		{
			population.insert(population.begin() + imid, make_pair(child, total_cost));
			return;
		}
		else if(total_cost > population[imid].second)
			imin = imid + 1;
		else
			imax = imid - 1;
	}
	population.insert(population.begin() + imin, make_pair(child, total_cost));
}

void Genetic::crossOver(vector<int>& parent1, vector<int>& parent2)
{
	vector<int> child1, child2;
	map<int, int> genes1, genes2;
	
	for(int i = 0; i < graph->V; i++)
	{
		genes1[parent1[i]] = 0;
		genes2[parent2[i]] = 0;
	}
	int point1 = rand() % (graph->V - 1) + 1;
	int point2 = rand() % (graph->V - point1) + point1;
	
	if(point1 == point2)
	{
		if(point1 - 1 > 1)
			point1--;
		else if(point2 + 1 < graph->V)
			point2++;
		else
		{
			int point = rand() % 10 + 1; 
			if(point <= 5)
				point1--;
			else
				point2++;
		}
	}
	for(int i = 0; i < point1; i++)
	{
		child1.push_back(parent1[i]);
		child2.push_back(parent2[i]);
		genes1[parent1[i]] = 1;
		genes2[parent2[i]] = 1;
	}
	for(int i = point2 + 1; i < graph->V; i++)
	{
		genes1[parent1[i]] = 1;
		genes2[parent2[i]] = 1;
	}	
	for(int i = point2; i >= point1; i--)
	{
		if(genes1[parent2[i]] == 0) 
		{
			child1.push_back(parent2[i]);
			genes1[parent2[i]] = 1; 	
		}
		else
		{
			for(map<int, int>::iterator it = genes1.begin(); it != genes1.end(); ++it)
			{
				if(it->second == 0) 
				{
					child1.push_back(it->first);
					genes1[it->first] = 1; 
					break; 
				}
			}
		}

		if(genes2[parent1[i]] == 0) 
		{
			child2.push_back(parent1[i]);
			genes2[parent1[i]] = 1; 
		}
		else
		{
			for(map<int, int>::iterator it = genes2.begin(); it != genes2.end(); ++it)
			{
				if(it->second == 0) 
				{
					child2.push_back(it->first);
					genes2[it->first] = 1; 
					break; 
				}
			}
		}
	}
	for(int i = point2 + 1; i < graph->V; i++)
	{
		child1.push_back(parent1[i]);
		child2.push_back(parent2[i]);
	}	
	int mutation = rand() % 100 + 1; 
	if(mutation <= mutation_rate) 
	{
		int index_gene1, index_gene2;
		index_gene1 = rand() % (graph->V - 1) + 1;
		index_gene2 = rand() % (graph->V - 1) + 1;
		
		swap(child1[index_gene2], child1[index_gene1]);
		swap(child2[index_gene1], child2[index_gene2]);
	}
	
	float total_cost_child1 = isValidSolution(child1);
	float total_cost_child2 = isValidSolution(child2);
	if(!existsChromosome(child1))
	{
		insertBinarySearch(child1, total_cost_child1); 
		real_size_population++; 
	}
	if(!existsChromosome(child2))
	{
		insertBinarySearch(child2, total_cost_child2); 
		real_size_population++; 
	}
}

void Genetic::run()
{
	initialPopulation(); 
	if(real_size_population == 0)
		return;
	
	float best = population[0].second;
	//for(int i = 0; i < generations; i++)
	int i=0;
	while(i>=0)
	{
		int  old_size_population = real_size_population;
		if(real_size_population >= 2)
		{	
			if(real_size_population == 2)
			{
				crossOver(population[0].first, population[1].first);
			}
			else
			{
				int parent1, parent2;
				do
				{
					parent1 = rand() % real_size_population;
					parent2 = rand() % real_size_population;
				}while(parent1 == parent2);
				crossOver(population[parent1].first, population[parent2].first);
			}
			int diff_population = real_size_population - old_size_population;
			
			if(diff_population == 2)
			{
				if(real_size_population > size_population)
				{
					population.pop_back();
					population.pop_back();
					
					real_size_population -= 2;
				}
			}
			else if(diff_population == 1)
			{
				if(real_size_population > size_population)
				{
					population.pop_back(); 
					real_size_population--; 
				}
			}
		} 
		else 
		{
			crossOver(population[0].first, population[0].first);
			if(real_size_population > size_population)
			{
				population.pop_back(); 
				real_size_population--; 
			}
		}
		if(i==0)
		{
			best = population[0].second;
			const vector<int>& vec = population[0].first;
			cout << "Tour : " << endl;
			for(int i = 0; i < graph->V; i++)
				cout << vec[i] << " ";
			cout << " | Cost: " << population[0].second << endl;;
		}
		else if(i%1000==0 && best > population[0].second)
		{
			best = population[0].second;
			const vector<int>& vec = population[0].first;
			cout << "Tour : " << endl;
			for(int i = 0; i < graph->V; i++)
				cout << vec[i] << " ";
			cout << " | Cost: " << population[0].second << endl;
		}
		i++;	
	}
	
	cout << "\nBest solution: " << endl;
	const vector<int>& vec = population[0].first;
	for(int i = 0; i < graph->V; i++)
		cout << vec[i] << " ";
	cout << " | Cost: " << population[0].second << endl;
}
