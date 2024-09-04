#include <bits/stdc++.h>
#include <vector>
#include <string>
#include <algorithm>

#ifndef MYHEADER_HPP_
#define MYHEADER_HPP_
/* myHeader.hpp */

void test4List();

// Definition for singly-linked list.
struct ListNode
{
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// Definition for a binary tree node.
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution
{
private:
    /* data */
public:
    // utills function
    void printList(ListNode *head);
    void printVector(const std::vector<int>& vec);
    void printVector2D(std::vector<std::vector<int>>&);
    void printVectoString(const std::vector<std::string>& vec);
    void printVector2D(std::vector<std::vector<std::string>>&);

    // 1.
    std::vector<int> twoSum1(std::vector<int> &, int);
    std::vector<int> twoSum2(std::vector<int> &, int);
    //
    ListNode *addTwoNumbers(ListNode *, ListNode *);
    
    //
    double findMedianSortedArrays(std::vector<int> &, std::vector<int> &);
    double findMedianSortedArrays_2(std::vector<int> &, std::vector<int> &);
    // 283.
    void moveZeroes(std::vector<int> &);
    void moveZeroes_1(std::vector<int> &);
    // 73.
    void sortColors(std::vector<int> &);
    // 160.
    ListNode *getIntersectionNode(ListNode *, ListNode *);
    ListNode *getIntersectionNode_1(ListNode *, ListNode *);
    // 206.
    ListNode *reverseList(ListNode *);
    ListNode *reverseList_1(ListNode *);
    ListNode *reverseList_2(ListNode *);
    // 234.
    bool isPalindrome(ListNode *);
    bool isPalindrome_1(ListNode *);
    bool isPalindrome_2(ListNode *);
    // 21.
    ListNode *mergeTwoLists(ListNode *, ListNode *);
    ListNode *mergeTwoLists_1(ListNode *, ListNode *);
    // 300.
    int lengthOfLIS(std::vector<int> &);
    // 80.
    int removeDuplicates(std::vector<int> &);
    // 蓝桥杯。回溯算法
    std::vector<int> pathPuzzle(int col[], int row[], int n);
    // 141.
    bool hasCycle(ListNode *head);
    bool hasCycle_1(ListNode *head);
    // 142.
    ListNode *detectCycle(ListNode *head);
    // 94.
    std::vector<int> inorderTraversal(TreeNode *root);
    void printTree(TreeNode *);
    // 104.二叉树最大深度
    int maxDepth(TreeNode *root);
    // 226.反转二叉树
    TreeNode *invertTree(TreeNode *);
    // 101.检查二叉树对称
    bool isSymmetric(TreeNode *root);
    // 543.二叉树最长直径
    int diameterOfBinaryTree(TreeNode *root);
    // 102.二叉树层序遍历
    std::vector<std::vector<int>> levelOrder(TreeNode *root);
    // 108.有序数组转为二叉搜索树
    TreeNode *sortedArrayToBST(std::vector<int> &nums);
    TreeNode *sortedArrayToBST_1(std::vector<int> &nums);
    // 35.
    int searchInsert(std::vector<int> &nums, int target);
    int searchInsert_1(std::vector<int> &nums, int target);
    // 20.有效括号
    bool isValid(std::string s);
    // 121.最大利润
    int maxProfit(std::vector<int> &prices);
    //70.爬楼梯
    int climbStairs(int n);
    int climbStairs_1(int n);
    int climbStairs_2(int n);
    //118.杨辉三角
    std::vector<std::vector<int>> generate(int numRows);
    //==========mid==========//
    //49.字母异位词 组合
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs);
    //128.最长连续序列长度
    int longestConsecutive(std::vector<int>& nums);

};

// 1.
void test4twoSum();
void test4moveZeroes();
// 73.
void test4sortColors();
// 160.
void test4getIntersectionNode();
// 206.
void test4reverseList();
// 234.
void test4isPalindrome();
// 21.
void test4mergeTwoLists();
// 300.
void test4lengthOfLIS();
// 80.
void test4removeDuplicates();
// 蓝桥杯。回溯算法
void test4pathPuzzle();
// 141.
void test4hasCycle();
// 142.
void test4detectCycle();
// 94.
void test4inorderTraversal();
// 104.
void test4maxDepth();
// 224.
void test4invertTree();
// 101.
void test4isSymmetric();
// 543.
void test4diameterOfBinaryTree();
// 102.
void test4levelOrder();
// 108.
void test4sortedArrayToBST();
// 35.
void test4searchInsert();
// 20.检查配对括号，循序
void test4isValid();
// 121.
void test4maxProfit();
//70.
void test4climbStairs();
//118.
void test4generate();
//=============mid============//
//49.
void test4groupAnagrams();
void test4longestConsecutive();









//=== My Test====
void myTest();

class MyHashSet
{
private:
    std::vector<std::list<int>> data;
    static const int base = 769;
    static int hash(int key)
    {
        return key % base;
    };

public:
    MyHashSet() : data(base) {}
    void displayHash();
    void add(int);
    void remove(int);
    bool contains(int);
    int hihi(int);
};
void test4MyHashSet();

class StringSolution
{
public:
    std::string longestPalindrome(std::string);
};
void test4StringSolution();

/* end of myHeader.hpp */
#endif /* MYHEADER_HPP_ */
