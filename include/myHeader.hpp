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

/// @brief solve vector can't be the element of hash
struct VectorHash
{
    size_t operator()(const std::vector<int> &v) const
    {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v)
        {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class Solution
{
private:
    /* hash vector */

public:
    // utills function
    void printList(ListNode *head);
    void printVector(const std::vector<int> &vec);
    void printVector2D(std::vector<std::vector<int>> &);
    void printVectorString(const std::vector<std::string> &vec);
    void printVector2D(std::vector<std::vector<std::string>> &);
    ListNode* create_list(ListNode* head, std::vector<int>);
    // 1.
    std::vector<int> twoSum1(std::vector<int> &, int);
    std::vector<int> twoSum2(std::vector<int> &, int);
    // 2. 两数相加
    ListNode *addTwoNumbers(ListNode *, ListNode *);
    ListNode* addTwoNumbers_1(ListNode* l1, ListNode* l2);
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
    // 70.爬楼梯
    int climbStairs(int n);
    int climbStairs_1(int n);
    int climbStairs_2(int n);
    // 118.杨辉三角
    std::vector<std::vector<int>> generate(int numRows);
    //==========mid==========//
    // 49.字母异位词 组合
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string> &strs);
    // 128.最长连续序列长度
    int longestConsecutive(std::vector<int> &nums);
    // 11.盛水最多
    int maxArea(std::vector<int> &height);
    // 15.三数之和为0的三元组
    std::vector<std::vector<int>> threeSum(std::vector<int> &nums);
    //3.无重复字串长度
    int lengthOfLongestSubstring(std::string s);
    //438. 找到字符串中所有字母异位词
    std::vector<int> findAnagrams(std::string s, std::string p);
    std::vector<int> findAnagrams_1(std::string s, std::string p);
    //560. 和为 K 的子数组
    int subarraySum(std::vector<int>& nums, int k);
    int subarraySum_1(std::vector<int>& nums, int k);
    //53. 最大子数组和
    int maxSubArray(std::vector<int>& nums);
    int maxSubArray_1(std::vector<int>& nums);
    //56. 合并区间
    std::vector<std::vector<int>> merge(std::vector<std::vector<int>>& intervals);
    //189. 轮转数组
    void rotate(std::vector<int>& nums, int k);
    void rotate_1(std::vector<int>& nums, int k);
    //238. 除自身以外数组的乘积
    std::vector<int> productExceptSelf(std::vector<int>& nums);
    //73. 矩阵置零
    void setZeroes(std::vector<std::vector<int>>& matrix);

    //334. 最大递增三元组
    bool increasingTriplet(std::vector<int>& nums);
    //返回最大递增三元组数值的值
    int getIncreasingTripletValue(std::vector<int>& nums);
    //54. 顺时针返回 m x n 矩阵的访问顺序
    std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix);
    //48. 顺时针旋转二维矩阵90度
    void rotate(std::vector<std::vector<int>>& matrix);
    //240. 搜索二维矩阵的target
    bool searchMatrix(std::vector<std::vector<int>>& matrix, int target);
    bool searchMatrix_1(std::vector<std::vector<int>>& matrix, int target);
    bool searchMatrix_2(std::vector<std::vector<int>>& matrix, int target);
    // 19. 删除链表的倒数第n个结点
    ListNode* removeNthFromEnd(ListNode* head, int n);
    ListNode* removeNthFromEnd_1(ListNode* head, int n);
    // 24. 交换链表中的结点（不能使用值交换）
    ListNode* swapPairs(ListNode* head);
    //==========hard==========//
    //42.
    int trap(std::vector<int>& height);
    int trap_1(std::vector<int>& height);//双指针
    int trap_2(std::vector<int>& height);//单调栈

    //239.滑动窗口最大值
    std::vector<int> maxSlidingWindow(std::vector<int>& nums, int k);
    std::vector<int> maxSlidingWindow_1(std::vector<int>& nums, int k);

    //76. 最小覆盖子串
    std::string minWindow(std::string s, std::string t);
    //41. 缺失的第一个正数
    int firstMissingPositive(std::vector<int>& nums);
    int firstMissingPositive_1(std::vector<int>& nums);
    
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
// 70.
void test4climbStairs();
// 118.
void test4generate();
// 56. 
void test4merge();

//=============mid============//
// 49.
void test4groupAnagrams();
void test4longestConsecutive();
void test4maxArea();
// 15.
void test4threeSum();
//1.
void test4lengthOfLongestSubstring();
//438. 找到字符串中所有字母异位词
void test4findAnagrams();
//560. 
void test4subarraySum();
//76. 
void test4minWindow();
//53.
void test4maxSubArray();
//189.
void test4rotate();
//238.
void test4productExceptSelf();
//73. 
void test4setZeroes();
//334. 
void test4increasingTriplet();
//54.
void test4spiralOrder();
//48.
void test4rotateMatrix();
//240.
void test4searchMatrix();
//2. 
void test4addTwoNumbers();
//19.
void test4removeNthFromEnd();
//24.
void test4swapPairs();





//=============hard============//

void test4trap();
void test4maxSlidingWindow();
void test4firstMissingPositive();

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
