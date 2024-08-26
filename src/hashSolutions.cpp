#include "../include/myHeader.hpp"

using namespace std;




void MyHashSet::add(int key){
    // 匹配 hash 值， not match push back.
    int h = hash(key);
    for (auto it = data[h].begin(); it != data[h].end(); it++){
        if(*it == key)
            return;   
    }
    data[h].push_back(key);
}

void MyHashSet::remove(int key){
    int h = hash(key);
    for (auto it = data[h].begin(); it != data[h].end(); it++){
        if(*it == key){
            // 传入指针
            data[h].erase(it);
            return;
        }
            
    }
}

bool MyHashSet::contains(int key){
    int h = hash(key);
    for (auto it = data[h].begin(); it != data[h].end(); it++){
        if(*it == key)
            return true;
    }
    return  false;
}
int MyHashSet::hihi(int k){
    return 1;
}

void MyHashSet::displayHash(){
    for(int i = 0; i < base; i ++){
        for(auto it = data[i].begin(); it != data[i].end(); it++){
            if (*it){
                cout << "第"<< i << "个buket, data contains:" << *it << endl;
            }
            
        }
    }
}


void test4MyHashSet(){
    MyHashSet myhash = MyHashSet();
    myhash.add(2);
    myhash.add(3);
    myhash.add(532);
    bool isexisit = myhash.contains(1);
    
    myhash.displayHash();
    
    cout << isexisit << endl;
}




