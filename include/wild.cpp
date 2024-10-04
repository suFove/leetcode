#include "../include/myHeader.hpp"

using namespace std;


int search(std::vector<std::pair<int, int>> work, int target){
    int left = 0;
    int right = work.size() - 1 ;
    int middle = 0;
    while(left <= right){
        middle = left + ((right - left) / 2);
        if(work[middle].first > target){
            right = middle - 1;
        }else if(work[middle].first < target){
            left = middle + 1;
        }else{
            return middle;
        }
    }
    return middle + 1; // 返回middle右侧
}   


int run(){
    int a, b;
    while (cin >> a >> b) { // 注意 while 处理多个 case
        int N = a, M = b;
        std::vector<std::pair<int, int>> work; // first is hard, sencond is pay
        std::vector<int> Ai; 
        int i = 0;
        while(i++ < N){
            int di, pi = 0;
            cin >> di >> pi;
            work.emplace_back(std::pair<int, int>(di, pi));
        }
        sort(work.begin(), work.end());
        int j, x = 0;

        string line, word;
        getline(cin, line);
        //将记录绑定到刚刚读取的行，用到了sstream里面的对象


        cin >> x;
        Ai.push_back(x);
        while (cin.get() != '\n') 
        {
            cin >> x;
            Ai.push_back(x);
        }
        //调用二分查找 查询first
        for(auto &a : Ai){
            int idx = search(work, a);
            cout << work[idx].second << endl;
        }
    }
}