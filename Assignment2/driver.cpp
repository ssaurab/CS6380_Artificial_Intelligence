#include "Othello.h"
#include "OthelloBoard.h"
#include "OthelloPlayer.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <list>
using namespace std;
using namespace Desdemona;


#define xyz 1000000

Turn my;

clock_t start,finish,end,s1,f1;
OthelloBoard globalBoard;

int movenum = 0;


bool canMove(char self, char opp, char *str)  {
	//if (str[0] != opp) return false;
    if (str[0] == opp){
	for (int ctr = 1; ctr < 8; ctr++) {
		if (str[ctr] == 'n') return false;
		if (str[ctr] == self) return true;
	}}
    else
        return false;
	return false;
}

bool isLegalMove(char self, char opp, char grid[8][8], int s_x, int s_y)   {
	if (grid[s_x][s_y] != 'n') return false;
	char str[10];
	int x, y, dx, dy, ctr;
	for (dy = -1; dy <= 1; dy++)
		for (dx = -1; dx <= 1; dx++)    {
	        // keep going if both velocities are zero
			if (!dy && !dx) continue;
			str[0] = '\0';
			for (ctr = 1; ctr < 8; ctr++)   {
				x = s_x + ctr*dx;
				y = s_y + ctr*dy;
				if (x >= 0 && y >= 0 && x<8 && y<8) str[ctr-1] = grid[x][y];
				else str[ctr-1] = 0;
			}
			if (canMove(self, opp, str)) return true;
		}
	return false;
}

int numValidMoves(char self, char opp, char grid[8][8])   {
	int count = 0, i, j;
	for(i = 0; i < 8; i++) for(j = 0; j < 8; j++) if(isLegalMove(self, opp, grid, i, j)) count++;
	return count;
}


//the heuristics were inspired from an online source, the weights were then fine tuned as per strategy requirements as needed
double othelloBoardEvaluator(char grid[8][8], char flag)  {

   

	char my_color = 'x',opp_color = 'o';
    int myTiles = 0, oppTiles = 0, i, j, k, myFrontTiles = 0, oppFrontTiles = 0, x, y;
    double p = 0.0, c = 0.0, l = 0.0, m = 0.0, f = 0.0, d = 0.0;

    int X1[] = {-1, -1, 0, 1, 1, 1, 0, -1};
    int Y1[] = {0, 1, 1, 1, 0, -1, -1, -1};

    int V[8][8] = 	{{ 20, -3, 11, 8 , 8, 11, -3, 20 },
    				{ -3, -7, -4, 1 , 1, -4, -7, -3 },
    				{ 11, -4, 2 , 2 , 2,  2, -4, 11 },
    				{ 8 , 1 , 2 , -3,-3,  2,  1, 8 },
    				{ 8 , 1 , 2 , -3,-3,  2,  1, 8 },
    				{ 11, -4, 2 , 2 , 2,  2, -4, 11 },
    				{ -3, -7, -4, 1 , 1, -4, -7, -3 },
    				{ 20, -3, 11, 8 , 8, 11, -3, 20 } };

	// Piece difference, frontier disks and disk squares
    for(i = 0; i < 8; i++)
        for(j = 0; j < 8; j++)  {
            if(grid[i][j] == my_color)  {
                d += V[i][j];
                myTiles++;
            } 
            else if(grid[i][j] == opp_color)  {
                d -= V[i][j];
                oppTiles++;
            }
            if(grid[i][j] != 'n')   {
                for(k = 0; k < 8; k++)  {
                    x = i + X1[k]; y = j + Y1[k];
                    if(x >= 0 && x < 8 && y >= 0 && y < 8 && grid[x][y] == 'n') {
                        if(grid[i][j] == my_color)  myFrontTiles++;
                        else oppFrontTiles++;
                        break;
                    }
                }
            }
        }

	if(movenum >= 30 && flag == 'l')
	{
		if(numValidMoves(my_color, opp_color, grid) == 0 && numValidMoves(opp_color, my_color, grid) == 0)
		{}
			///cout << "leaf yeah" << endl;
		//else
			//cout << "not a leaf sir!" << endl;
		//return myTiles - oppTiles;
	}
        // int V[8][8]={ 10,-3,2,2,2,2,-3,10
    //             ,-3,-4,-1,-1,-1,-1,-4,-3
    //             ,2,-1,1,0,0,1,-1,-2
    //             ,2,-1,0,1,1,0,-1,2
    //             ,2,-1,0,1,1,0,-1,2
    //             ,2,-1,1,0,0,1,1,-2
    //             ,-3,-4,-1,-1,-1,1,-4,-3
    //             ,10,-3,2,2,2,2,-3,10 };

    if(myTiles > oppTiles) p = (100.0 * myTiles)/(myTiles + oppTiles);
    else if(myTiles < oppTiles) p = -(100.0 * oppTiles)/(myTiles + oppTiles);

    if(myFrontTiles > oppFrontTiles) f = -(100.0 * myFrontTiles)/(myFrontTiles + oppFrontTiles);
    else if(myFrontTiles < oppFrontTiles) f = (100.0 * oppFrontTiles)/(myFrontTiles + oppFrontTiles);

    // Corner occupancy
    myTiles = oppTiles = 0;
    int a = 0;
    int b = 0;
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            if(grid[a][b] == my_color) myTiles++;
            else if (grid[a][b] == opp_color) oppTiles++;
            b = 7;}
        a = 7;
        b = 0;
    }


    /*if(grid[0][0] == my_color) myTiles++;
    else if(grid[0][0] == opp_color) oppTiles++;
    if(grid[0][7] == my_color) myTiles++;
    else if(grid[0][7] == opp_color) oppTiles++;
    if(grid[7][0] == my_color) myTiles++;
    else if(grid[7][0] == opp_color) oppTiles++;
    if(grid[7][7] == my_color) myTiles++;
    else if(grid[7][7] == opp_color) oppTiles++;*/
    c = 25 * (myTiles - oppTiles);

    // Corner closeness
    myTiles = oppTiles = 0;
    if(grid[0][0] == 'n')   {
        if(grid[0][1] == my_color) myTiles++;
        else if(grid[0][1] == opp_color) oppTiles++;
        if(grid[1][1] == my_color) myTiles++;
        else if(grid[1][1] == opp_color) oppTiles++;
        if(grid[1][0] == my_color) myTiles++;
        else if(grid[1][0] == opp_color) oppTiles++;
    }
    if(grid[0][7] == 'n')   {
        if(grid[0][6] == my_color) myTiles++;
        else if(grid[0][6] == opp_color) oppTiles++;
        if(grid[1][6] == my_color) myTiles++;
        else if(grid[1][6] == opp_color) oppTiles++;
        if(grid[1][7] == my_color) myTiles++;
        else if(grid[1][7] == opp_color) oppTiles++;
    }
    if(grid[7][0] == 'n')   {
        if(grid[7][1] == my_color) myTiles++;
        else if(grid[7][1] == opp_color) oppTiles++;
        if(grid[6][1] == my_color) myTiles++;
        else if(grid[6][1] == opp_color) oppTiles++;
        if(grid[6][0] == my_color) myTiles++;
        else if(grid[6][0] == opp_color) oppTiles++;
    }
    if(grid[7][7] == 'n')   {
        if(grid[6][7] == my_color) myTiles++;
        else if(grid[6][7] == opp_color) oppTiles++;
        if(grid[6][6] == my_color) myTiles++;
        else if(grid[6][6] == opp_color) oppTiles++;
        if(grid[7][6] == my_color) myTiles++;
        else if(grid[7][6] == opp_color) oppTiles++;
    }
    l = -10 * (myTiles - oppTiles);

    // Mobility
    myTiles = numValidMoves(my_color, opp_color, grid);
    oppTiles = numValidMoves(opp_color, my_color, grid);
    if(myTiles > oppTiles) m = (100.0 * myTiles)/(myTiles + oppTiles);
    else if(myTiles < oppTiles) m = -(100.0 * oppTiles)/(myTiles + oppTiles);

    // final weighted score

	if(movenum < 22)
	{
		return (11 * p) + (950* c) + (420 * l) + (125 * m) + (65 * f) + (11 * d);
	}

   

    double score = (13 * p) + (900* c) + (400 * l) + (110 * m) + (78.396 * f) + (11 * d);
    return score;
}

double alphabeta(OthelloBoard board, Move move, Turn turn, short level, double alpha, double beta) {
    finish = clock();
    if(((double)(finish-start)/CLOCKS_PER_SEC)>1.95) {
        if(level%2 == 1) return -xyz;
        return xyz;
    }
	if(level == 6) {
		char grid[8][8];
		for(int i=0;i<8;i++) {
			for(int j=0;j<8;j++) {
				Coin findTurn = board.get(i,j);
				if(findTurn == turn) grid[i][j] = 'o';
				else if(findTurn == other(turn)) grid[i][j] = 'x';
				else grid[i][j] = 'n';
			}
		}
		return othelloBoardEvaluator(grid,'l');
	}
	board.makeMove(turn,move);
	turn = other(turn);
	list<Move> newMoves = board.getValidMoves(turn);
	list<Move>::iterator iter;
	double range_decide = -xyz;
	if((level%2 == 1)) range_decide *= -1;
	if(!(newMoves.size())) {

		int coindiff = 0;

		for(int i=0;i<8;i++) {
			for(int j=0;j<8;j++) {
				Coin findTurn = board.get(i,j);
				if(findTurn == other(turn))coindiff++;
				else if(findTurn == turn) coindiff--;
			}
		}
		
		if(coindiff == 0) coindiff --;
		//cout << "range_decide updated to : " << range_decide * coindiff << endl;



        //we take 1000000 value to and multiply with coin diff to escape so as to disgtinguish between
        //the goodness of leaves

		range_decide *= coindiff; 
		return range_decide;

	}
	for( iter = newMoves.begin(); iter!=newMoves.end(); iter++) {

		double current_treshold = alphabeta(board,*iter,turn,level+1,alpha,beta);

		if(level%2 == 1) {
			range_decide = min(range_decide,current_treshold);
			beta = min(beta,range_decide);
		}
		else {
			range_decide = max(range_decide,current_treshold);
			alpha = max(alpha,range_decide);		
		}
		if(beta<=alpha) break;
	}
	return range_decide; 
}

double tester(OthelloBoard board,Turn turn) {

    char grid[8][8];

    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
        Coin findTurn = board.get(i,j);
        
        //this stands for the grid that the eval function asks for, 
        //here : 
        // x : X - our coin
        // o : O - the opponents coin
        // n : N - null aka empty
        if(findTurn == turn) grid[i][j] = 'x';

        else if(findTurn == other(turn)) grid[i][j] = 'o';
        
        else grid[i][j] = 'n';
        }
    }

    return othelloBoardEvaluator(grid,'i');
}

bool eval_functor(Move a, Move b) {

    OthelloBoard cache_board_left = globalBoard;
    OthelloBoard cache_board_right = globalBoard;

    cache_board_left.makeMove(my,a);
    cache_board_right.makeMove(my,b);
    
    bool flag = false;
    if(tester(cache_board_left,my) > tester(cache_board_right,my))
        return true;
    return flag;

}

class MyBot: public OthelloPlayer
{
    public:
        /**
         * Initialisation routines here
         * This could do anything from open up a cache of "best moves" to
         * spawning a background processing thread. 
         */
        MyBot( Turn turn );

        /**
         * Play something 
         */
        virtual Move play( const OthelloBoard& board );
    private:
};

MyBot::MyBot( Turn turn )
    : OthelloPlayer( turn )
{
}

Move MyBot::play( const OthelloBoard& board )
{
    movenum++;
    start = clock();
    list<Move> moves = board.getValidMoves( turn );
    my = turn;
    globalBoard = board;

    moves.sort(eval_functor);
    
    list<Move>::iterator it,firstref = moves.begin();
    
    //first we sort the function based on the eval values of the nodes immediately below and then
    //we sweep the tree from left to right as much as thetime permits
    //this step corresponds to assuming that the best evaluated level-1 move is the ultimate best move
    Move bestMove((*firstref).x,(*firstref).y);
    
    double retVal = -xyz;
    double MAX = xyz, MIN = -xyz;

    OthelloBoard copyBoard = board;
    
    short level = 1;
    
    for(it = moves.begin(); it!=moves.end(); it++) {

    	double currValue = alphabeta(copyBoard,*it,turn,level,MIN,MAX);

    	if(currValue > retVal) {
    		retVal = currValue;
    		bestMove = *it;
    	}
        //we take 1000000 value to and multiply with coin diff to escape so as to disgtinguish between
        //the goodness of leaves
	   //cout << "eval : "<<fixed << currValue << endl;
    	copyBoard = board;
    }
    finish = clock();
    //this is to measure the time taken for the the program from start to returning a move to he game server
    //cout << endl << "finished at : " << (double)(finish - start)/(CLOCKS_PER_SEC)  << endl;
    return bestMove;
}

// The following lines are _very_ important to create a bot module for Desdemona

extern "C" {
    OthelloPlayer* createBot( Turn turn )
    {
        return new MyBot( turn );
    }

    void destroyBot( OthelloPlayer* bot )
    {
        delete bot;
    }
}
