#include "../include/myHeader.hpp"

using namespace std;

// utills function

void Solution::printList(ListNode *head)
{
    while (head)
    {
        cout << head->val << " -> ";
        head = head->next;
    }
    cout << "nullptr" << endl;
}

void Solution::printVector(const vector<int> &vec)
{
    cout << " [ ";
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << " ";
    }
    cout << "]" << endl;
}

void Solution::printVector2D(vector<vector<int>> &res)
{
    cout << "[" << endl;
    for (const auto &r : res)
    {
        printVector(r);
    }
    cout << "]" << endl;
}

void Solution::printVectoString(const vector<std::string> &vec)
{
    cout << " [ ";
    for (const auto &s : vec)
    {
        cout << "\"" << s << "\"" << " ";
    }
    cout << "]" << endl;
}

void Solution::printVector2D(vector<vector<string>> &res)
{
    cout << "[" << endl;
    for (const auto &r : res)
    {
        printVectoString(r);
    }
    cout << "]" << endl;
}

//======= easy =========//
// 1.两数之和
vector<int> Solution::twoSum1(vector<int> &nums, int target)
{
    if (2 > nums.size())
    {
        return vector<int>{};
    }
    if (2 == nums.size())
    {
        return vector<int>{0, 1};
    }
    else
    {
        for (int i = 0; i < nums.size() - 1; i++)
        {
            for (int j = i + 1; j < nums.size(); j++)
            {
                if (nums[i] + nums[j] == target)
                    return vector<int>{i, j};
            }
        }
        return vector<int>{};
    }
}

vector<int> Solution::twoSum2(vector<int> &nums, int target)
{
    unordered_map<int, int> map;
    for (int i = 0; i < nums.size(); ++i)
    {
        if (map.count(target - nums[i]))
        {
            return {map[target - nums[i]], i};
        }
        map[nums[i]] = i;
    }
    return {-1, -1};
}

void test4twoSum()
{
    Solution ss;
    int target = 9;
    vector<int> nums = {2, 7, 11, 15};
    vector<int> res = ss.twoSum1(nums, target);
    if (!res.empty())
        for (auto ele : res)
            cout << ele << endl;
}

// 1.两数之和
ListNode *Solution::addTwoNumbers(ListNode *l1, ListNode *l2)
{
    ListNode *dummy = new ListNode();
    ListNode *current = dummy;
    int carry = 0;

    while (l1 || l2 || carry)
    {
        int sum = carry;
        if (l1)
        {
            sum += l1->val;
            l1 = l1->next;
        }
        if (l2)
        {
            sum += l2->val;
            l2 = l2->next;
        }

        carry = sum / 10;
        current->next = new ListNode(sum % 10);
        current = current->next;
    }

    return dummy->next;
}

// 4.寻找两个正序数组的中位数
// （归并）合并两个数组，找中位数
double Solution::findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2)
{
    double ans = 0.0;
    vector<int> vec;
    // 判空 nums1
    if (nums1.empty())
    {
        // 处理nums2
        if (nums2.size() & 1 == 1)
        { // 奇数
            ans = nums2[nums2.size() / 2];
        }
        else
        {
            ans = (nums2[nums2.size() / 2] + nums2[nums2.size() / 2 - 1]) / 2.0;
        }
    }
    // 判空nums2
    if (nums2.empty())
    {
        // 处理nums1
        if (nums1.size() & 1 == 1)
        { // 奇数
            ans = nums1[nums1.size() / 2];
        }
        else
        {
            ans = (nums1[nums1.size() / 2] + nums1[nums1.size() / 2 - 1]) / 2.0;
        }
    }

    // 非空
    int i = 0;
    int j = 0;
    while (i + j < nums1.size() + nums2.size())
    {
        if (i == nums1.size())
        { // num1 用完
            while (j < nums2.size())
            {
                vec.insert(vec.end(), nums2[j++]);
            }
            break;
        }
        if (j == nums2.size())
        { // nums2 用完
            while (i < nums1.size())
            {
                vec.insert(vec.end(), nums1[i++]);
            }
            break;
        }

        // 都没用完,选小的
        if (nums1[i] < nums2[j])
        {
            vec.insert(vec.end(), nums1[i++]);
        }
        else
        {
            vec.insert(vec.end(), nums2[j++]);
        }

    } // while

    // 处理 ans
    if (vec.size() & 1 == 1)
    { // 奇数
        ans = vec[vec.size() / 2];
    }
    else
    {
        ans = (vec[vec.size() / 2] + vec[vec.size() / 2 - 1]) / 2.0;
    }

    return ans;
}
// 4.寻找两个正序数组的中位数
/*
    考虑单循环条件，不需要真的合并数组，
    2个数组均为有序，递增。至多需要len/2+1次遍历即可得到各自中位数，合并后亦是
    考虑奇偶数，需要2个变量。
    考虑循环条件。
*/
double Solution::findMedianSortedArrays_2(vector<int> &nums1, vector<int> &nums2)
{
    double ans = 0.0;
    int old, curr = -1;     // 循环保存变量
    int aStart, bStart = 0; // 两个数组的开始下标
    for (int i = 0; i <= (nums1.size() + nums2.size()) / 2; i++)
    {
        old = curr;
        // A数组还有，并且列举B数组的所有条件{B: 用完，或者A小于B}
        if (aStart < nums1.size() && (bStart >= nums2.size() || nums1[aStart] <= nums2[bStart]))
        {
            curr = nums1[aStart++];
        }
        else
        {
            curr = nums2[bStart++];
        }
    } // for
    if ((nums1.size() + nums2.size()) & 1 == 1)
    { // 奇数
        ans = curr;
    }
    else
    {
        ans = (old + curr) / 2.0;
    }

    return ans;
}

void test4List()
{
    Solution ss;
    // // Create two linked lists to represent numbers
    // ListNode* l1 = new ListNode(2, new ListNode(4, new ListNode(3)));
    // ListNode* l2 = new ListNode(5, new ListNode(6, new ListNode(4)));

    // // Add the two numbers and print the result
    // ListNode* result = ss.addTwoNumbers(l1, l2);
    // ss.printList(result);

    vector<int> nums1 = {1, 0, 5, 0, 44};
    vector<int> nums2 = {1, 7};
    // double ans = ss.findMedianSortedArrays(nums1, nums2);
    double ans = ss.findMedianSortedArrays_2(nums1, nums2);
    // for(auto s: st){
    //     cout << s << endl;
    // }
    cout << ans << endl;
}

// 283.移动0到数组最后
void Solution::moveZeroes(vector<int> &nums)
{
    int counter = 0;
    for (int i = 0; i < nums.size() - counter; ++i)
    {
        if (nums[i] == 0)
        {
            for (int j = i; j < nums.size() - 1 - counter; ++j)
            {
                nums[j] = nums[j + 1];
            }
            nums[nums.size() - 1 - counter] = 0;
            ++counter;
            --i;
        }
    }
}

// 双指针
void Solution::moveZeroes_1(vector<int> &nums)
{
    int slow = 0;
    // 双指针，前面遇到0，与后面交换
    for (int fast = 0; fast < nums.size(); fast++)
    {
        if (nums[fast] != 0)
        {
            swap(nums[slow++], nums[fast]);
        }
        // else:
        // fast++;
    }
}

void test4moveZeroes()
{
    vector<int> nums1 = {1, 0, 5, 0, 0, 44};
    // vector<int> nums1 = {0};
    Solution ss;
    // ss.moveZeroes(nums1);
    ss.moveZeroes_1(nums1);
    for (auto s : nums1)
        cout << s << endl;
}

// 颜色分类：0，1，2数组，顺序排序
void Solution::sortColors(vector<int> &nums)
{
    int zero = 0;
    int two = nums.size() - 1;
    for (int i = 0; i <= two; i++)
    {
        // if(nums[i] == 2){
        //     swap(nums[i], nums[two--]);
        // }
        // 为什么不能用if, 要用while?
        // 【2，1，2】避免2与末尾2交换，保证two上为非2元素
        while (nums[i] == 2 && i <= two)
        {
            swap(nums[i], nums[two--]);
        }
        if (nums[i] == 0)
        {
            swap(nums[i], nums[zero++]);
        }
    }
}

void test4sortColors()
{
    vector<int> nums1 = {2, 0, 2, 1, 1, 0};
    // vector<int> nums1 = {0};
    Solution ss;
    // ss.moveZeroes(nums1);
    ss.sortColors(nums1);
    for (auto s : nums1)
        cout << s << endl;
}

// 链表检索交叉结点
ListNode *Solution::getIntersectionNode(ListNode *headA, ListNode *headB)
{
    ListNode *pa = headA;
    // ListNode* pb =  headB;
    while (pa)
    {
        ListNode *pb = headB;
        while (pb)
        {
            if (pa == pb)
                return pb;
            pb = pb->next;
        }
        pa = pa->next;
    }

    return NULL;
}

ListNode *Solution::getIntersectionNode_1(ListNode *headA, ListNode *headB)
{
    // 2个有一个为空
    if (!headA || !headB)
        return NULL;

    ListNode *pa = headA, *pb = headB;
    while (pa != pb)
    {
        pa = pa ? pa->next : headB;
        pb = pb ? pb->next : headA;
    }
    return pa;
}

void test4getIntersectionNode()
{
    Solution ss;
    ListNode *IntersectionNode = new ListNode(4, new ListNode(3));
    ListNode *headA = new ListNode(2, IntersectionNode);
    ListNode *headB = new ListNode(5, IntersectionNode);

    // ListNode* res = ss.getIntersectionNode(headA, headB);
    ListNode *res = ss.getIntersectionNode_1(headA, headB);
    cout << res->val << endl;
}

// 206.反转链表
ListNode *Solution::reverseList(ListNode *head)
{
    ListNode *p = head;
    vector<int> stack;
    // 入栈
    while (p)
    {
        stack.push_back(p->val);
        p = p->next;
    }
    // 出栈
    p = head;
    while (p)
    {
        p->val = stack.back();
        stack.pop_back();
        p = p->next;
    }
    return head;
}

// 206.迭代反转链表
ListNode *Solution::reverseList_1(ListNode *head)
{
    // 需要三个指针，pre, cur, tmp;
    ListNode *pre = nullptr;
    ListNode *cur = head;
    ListNode *tmp = nullptr;
    while (cur)
    {
        tmp = cur->next;
        cur->next = pre;
        pre = cur;
        cur = tmp;
    }
    return pre;
}
// 206.递归反转链表
ListNode *Solution::reverseList_2(ListNode *head)
{
    // 定义递归匿名函数。递归 lambda 需要使用 std::function 进行包装
    function<ListNode *(ListNode * cur, ListNode * pre)> recur = [&](ListNode *cur, ListNode *pre) -> ListNode *
    {
        if (!cur)
            return pre;
        ListNode *res = recur(cur->next, cur);
        cur->next = pre;
        return res;
    };

    return recur(head, nullptr);
}

void test4reverseList()
{
    Solution ss;
    ListNode *IntersectionNode = new ListNode(4, new ListNode(3));
    ListNode *headA = new ListNode(2, IntersectionNode);
    cout << "original list:";
    ss.printList(headA);
    cout << endl;
    ListNode *res = ss.reverseList_2(headA);

    cout << "reversed list:";
    ss.printList(res);
    cout << endl;
}

// 234.
bool Solution::isPalindrome(ListNode *head)
{
    if (!head->next)
        return true;

    vector<int> vec;
    while (head)
    {
        vec.emplace_back(head->val);
        head = head->next;
    }
    int i = 0, j = vec.size() - 1;
    for (; i < j; i++, j--)
    {
        if (vec[i] != vec[j])
            return false;
    }
    return true;
}

// 234. 递归链表回文
bool Solution::isPalindrome_1(ListNode *head)
{
    // 记录head
    ListNode *p = head;
    // 定义匿名函数 使用function包装。
    function<bool((ListNode * cur))> recur = [&](ListNode *cur) -> bool
    {
        // 结束条件：if(!cur)
        // 前进条件：
        if (cur)
        { // 内容不为空
            // 结束条件:从stack上面往下看
            if (!recur(cur->next)) // 2. 拿到底层结果，!true, 跳过if
                return false;
            if (cur->val != p->val) // 3. 当前判断结果，返回上一层
                return false;
            p = p->next;
        }
        return true; // 1.最后一个结点返回true
    };
    return recur(head);
}

// 234. 递归链表回文
bool Solution::isPalindrome_2(ListNode *head)
{

    // 匿名函数；找到中间结点，反转后半部分。
    function<ListNode *(ListNode *)> getHalfTailNode = [&](ListNode *head) -> ListNode *
    {
        // 快慢指针；
        // fast一次移动2个结点，走到最后的时候，slow刚到到达list中间结点。
        ListNode *fast = head;
        ListNode *slow = head;
        // ListNode* slow = head;
        while (fast->next != nullptr && fast->next->next != nullptr)
        {
            fast = fast->next->next;
            slow = slow->next;
        }
        return slow;
    };
    // 匿名函数：反转链表
    function<ListNode *(ListNode *)> reverseList = [&](ListNode *head) -> ListNode *
    {
        ListNode *cur = head;
        ListNode *pre = nullptr;
        ListNode *tmp = nullptr;
        while (cur)
        {
            tmp = cur->next;
            cur->next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    };
    // ------逻辑开始------
    if (head == nullptr)
    {
        return true;
    }
    ListNode *halfNode = getHalfTailNode(head);
    ListNode *halfTailReversed = reverseList(halfNode->next);

    ListNode *p1 = head;
    ListNode *p2 = halfTailReversed;
    bool res = true;
    while (res && p2)
    {
        if (p1->val != p2->val)
            res = false;
        p1 = p1->next;
        p2 = p2->next;
    }
    // 还原链表并返回结果
    halfNode->next = reverseList(halfTailReversed);
    return res;
}

void test4isPalindrome()
{
    // ListNode *headA = new ListNode(1, new ListNode(1, new ListNode(2, new ListNode(1))));
    ListNode *headA = new ListNode(2, new ListNode(3, new ListNode(2)));
    Solution ss;
    ss.printList(headA);

    cout << ss.isPalindrome_2(headA) << endl;
}

// 21.合并有序链表,递归
ListNode *Solution::mergeTwoLists(ListNode *list1, ListNode *list2)
{
    if (list1 == nullptr)
        return list2;
    if (list2 == nullptr)
        return list1;

    if (list1->val < list2->val)
    {
        list1->next = Solution::mergeTwoLists(list1->next, list2);
        return list1;
    }
    else
    {
        list2->next = Solution::mergeTwoLists(list1, list2->next);
        return list2;
    }
}

ListNode *Solution::mergeTwoLists_1(ListNode *list1, ListNode *list2)
{
    ListNode head;
    ListNode *cur = &head;
    while (list1 != nullptr && list2 != nullptr)
    {
        if (list1->val <= list2->val)
        {
            cur->next = list1;
            list1 = list1->next;
        }
        else
        {
            cur->next = list2;
            list2 = list2->next;
        }
        cur = cur->next; // 移动
    }
    // 连接尾部
    cur->next = (list1 == nullptr) ? list2 : list1;
    return head.next;
}

void test4mergeTwoLists()
{
    Solution ss;
    ListNode *headA = new ListNode(1, new ListNode(3, new ListNode(5, new ListNode(6))));
    ListNode *headB = new ListNode(2, new ListNode(5, new ListNode(7)));
    ss.printList(headA);
    ss.printList(headB);

    ListNode *res = ss.mergeTwoLists_1(headA, headB);
    ss.printList(res);
}

// 300.最长递增子序列长度
int Solution::lengthOfLIS(vector<int> &nums)
{
    // 存储nums第一个小于lis
    vector<int> lis(nums.size(), INT_MAX);
    for (auto x : nums)
    {
        lis[lower_bound(lis.begin(), lis.end(), x) - lis.begin()] = x;
    }
    return lower_bound(lis.begin(), lis.end(), INT_MAX) - lis.begin();
}

void test4lengthOfLIS()
{
    vector<int> nums = {10, 9, 2, 5, 3, 7, 101, 18};
    Solution ss;
    int res = ss.lengthOfLIS(nums);
    cout << res << endl;
}

// 80.删除重复数字，最多保留2个，nums为升序数组
int Solution::removeDuplicates(std::vector<int> &nums)
{
    if (nums.size() <= 2)
    {
        return nums.size();
    }
    int i = 2;
    int ori = i - 1;
    for (; i < nums.size(); i++)
    {
        if (nums[i] == nums[ori] && nums[i] == nums[ori - 1])
        {
            continue;
        }
        else
        {
            nums[++ori] = nums[i];
        }
    }
    // for(int j = 0; j <= ori; j++){
    //     cout << nums[j];
    // }
    // cout << endl;
    return ori + 1;
}

void test4removeDuplicates()
{
    vector<int> nums = {1, 2, 2, 2, 3, 5, 5, 5, 5, 5, 5, 7};
    Solution ss;
    int res = ss.removeDuplicates(nums);
    cout << res << endl;
}

// 蓝桥杯路径之谜，回溯算法。
/**
 * 数独：输入一个nxn的矩阵，要求从左上角走到右下角，寻找路径，走过的格子数与行列的标记数一致。
 *
 *
 */
vector<int> Solution::pathPuzzle(int row[], int col[], int n)
{
    // 枚举下个状态
    int dx[4] = {0, 0, 1, -1};
    int dy[4] = {1, -1, 0, 0};
    // step1:开一个数组记录是否走过的状态,nxn
    vector<vector<bool>> vis(n, vector<bool>(n, false));
    // step2:开一个路径，记录
    vector<int> path;
    // step3:记录格子对应的下标, give坐标查询是否走过？
    function<bool(int, int)> dfs = [&](int x, int y) -> bool
    {
        // step4:枚举边界条件
        if (x < 0 || x >= n || y < 0 || y >= n)
            return false;
        if (vis[x][y])
            return false;
        if (row[x] == 0 || col[y] == 0)
            return false;
        // extra 剪枝
        if (row[x] == 1 && accumulate(row, row + x, 0) != 0)
            return false;
        if (col[x] == 1 && accumulate(col, col + x, 0) != 0)
            return false;
        // step5:更新状态
        row[x]--;
        col[y]--;
        vis[x][y] = true;
        path.push_back(x * n + y);
        // step6:结束状态, 走到右下角
        if (x == n - 1 && y == n - 1 &&
            accumulate(row, row + n, 0) &&
            accumulate(col, col + n, 0))
            return true;
        // step7:状态转移
        for (int d = 0; d < 4; d++)
        {
            if (dfs(x + dx[d], y + dy[d]))
                return true;
        }
        // step8:还原状态，与更新相反
        row[x]++;
        col[y]++;
        vis[x][y] = false;
        path.pop_back();
        return false;
    };
    dfs(0, 0);
    return path;
}

void test4pathPuzzle()
{
    int row[4] = {2, 4, 3, 4};
    int col[4] = {4, 3, 3, 3};
    int n = 4;
    // Output: [0, 1, 2, 3]
    Solution ss;
    vector<int> res = ss.pathPuzzle(row, col, n);
    for (auto x : res)
    {
        cout << x << ' ';
    }
}

// 141.判断循环链表，返回位置pos下标
bool Solution::hasCycle(ListNode *head)
{
    // 创建HashSet
    unordered_set<ListNode *> seen;
    ListNode *p = head;
    while (p)
    {
        if (seen.count(p))
            return true;
        else
        {
            seen.insert(p);
            p = p->next;
        }
    }
    return false;
}
// 141.判断循环链表，返回位置pos下标
bool Solution::hasCycle_1(ListNode *head)
{
    if (head == nullptr || head->next == nullptr)
        return false;
    // 快慢指针
    ListNode *slow = head;
    ListNode *fast = head->next;
    while (fast != slow)
    {
        if (fast == nullptr || fast->next == nullptr)
        {
            return false;
        }
        slow = slow->next;
        fast = fast->next->next;
    }
    return true;
}

void test4hasCycle()
{

    ListNode *headA = new ListNode(3);
    ListNode *headB = new ListNode(4, headA);
    headA->next = headB;
    Solution ss;
    bool res = ss.hasCycle_1(headB);
    cout << res;
}

// 142.检测环形链表的结点位置
ListNode *Solution::detectCycle(ListNode *head)
{
    unordered_set<ListNode *> seen;

    for (ListNode *p = head; p; p = p->next)
    {
        if (seen.find(p) != seen.end())
        {
            return p;
        }
        seen.insert(p);
        p = p->next;
    }
    return NULL;
}

void test4detectCycle()
{
    ListNode *headA = new ListNode(3);
    ListNode *headB = new ListNode(4, headA);
    headA->next = headB;
    Solution ss;
    ListNode *res = ss.detectCycle(headB);
    cout << res->val;
}

// 94.二叉树中序遍历
std::vector<int> Solution::inorderTraversal(TreeNode *root)
{
    // 顺序：左根右
    vector<int> seq;

    function<void(TreeNode *, vector<int> &)> ldr = [&](TreeNode *node, vector<int> &res) -> void
    {
        if (node == nullptr)
        {
            return;
        }
        ldr(node->left, res);
        res.push_back(node->val);
        ldr(node->right, res);
    };

    ldr(root, seq);

    return seq;
}

void Solution::printTree(TreeNode *root)
{
    cout << endl;
    function<void(TreeNode *, int)> inorder = [&](TreeNode *root, int indent = 0)
    {
        if (root == nullptr)
            return;
        inorder(root->right, indent + 4);
        cout << string(indent, ' ') << root->val << "\n";
        inorder(root->left, indent + 4);
    };
    inorder(root, 0);
    cout << "-root------>------leaf-" << endl;
}

void test4inorderTraversal()
{
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);

    Solution ss;
    cout << endl;
    vector<int> res = ss.inorderTraversal(root);
    for (auto x : res)
    {
        cout << x << ' ';
    }
    ss.printTree(root);
}

int Solution::maxDepth(TreeNode *root)
{
    int depth = 0;

    function<int(TreeNode *)> hight = [&](TreeNode *root) -> int
    {
        if (root == nullptr)
            return 0;

        int left = hight(root->left);
        int right = hight(root->right);
        return left > right ? left + 1 : right + 1;
    };

    depth = hight(root);
    return depth;
}

void test4maxDepth()
{
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);
    Solution ss;
    int res = ss.maxDepth(root);
    cout << res << endl;
}

TreeNode *Solution::invertTree(TreeNode *root)
{

    function<TreeNode *(TreeNode *)> invert = [&](TreeNode *root) -> TreeNode *
    {
        if (root == nullptr)
            return nullptr;
        // 交换左右子树
        swap(root->left, root->right);
        // 递归左子树：
        invert(root->left);
        invert(root->right);
        return root;
    };

    return invert(root);
}

void test4invertTree()
{
    TreeNode *root = new TreeNode(4, new TreeNode(2, new TreeNode(1), new TreeNode(3)), new TreeNode(7, new TreeNode(6), new TreeNode(9)));
    Solution ss;
    vector<int> original_res = ss.inorderTraversal(root);
    ss.invertTree(root);
    vector<int> precced_res = ss.inorderTraversal(root);

    cout << "oiginal res: ";
    for (auto x : original_res)
    {
        cout << x << ' ';
    }
    cout << endl
         << "preceed res: ";
    for (auto x : precced_res)
    {
        cout << x << ' ';
    }

    ss.printTree(root);
}

bool Solution::isSymmetric(TreeNode *root)
{

    function<bool(TreeNode *, TreeNode *)> recurr = [&](TreeNode *left, TreeNode *right) -> bool
    {
        if (left == nullptr && right == nullptr)
            return true;
        if (left == nullptr || right == nullptr || left->val != right->val)
            return false;

        return recurr(left->left, right->right) && recurr(left->right, right->left);
    };
    return root == nullptr || recurr(root->left, root->right);
}

void test4isSymmetric()
{
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(2);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(4);
    root->right->right = new TreeNode(3);
    Solution ss;
    ss.printTree(root);
    bool res = ss.isSymmetric(root);
    cout << res << endl;
}

// 543.二叉树最长直径(任意2结点)
int Solution::diameterOfBinaryTree(TreeNode *root)
{
    // 计算左孩子深度+右孩子深度+1(根)
    int ans = 0;
    function<int(TreeNode *)> depth = [&](TreeNode *root) -> int
    {
        if (root == nullptr)
            return 0;
        int left = depth(root->left);
        int right = depth(root->right);
        ans = max(ans, left + right);
        return max(left, right) + 1;
    };
    depth(root);
    return ans;
}

void test4diameterOfBinaryTree()
{
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(2);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(5);
    // root->right->left = new TreeNode(4);
    root->right->right = new TreeNode(3);
    Solution ss;
    ss.printTree(root);
    int res = ss.diameterOfBinaryTree(root);
    cout << res << endl;
}

vector<vector<int>> Solution::levelOrder(TreeNode *root)
{
    vector<vector<int>> res;
    if (root == nullptr)
    {
        return res;
    }

    queue<TreeNode *> que;
    que.push(root);
    while (!que.empty())
    {
        // 需要保留层
        int cur_level_size = que.size();
        res.push_back(vector<int>()); // 扔进去空数组
        for (int i = 0; i < cur_level_size; ++i)
        {
            TreeNode *cur = que.front();
            que.pop();
            res.back().push_back(cur->val);
            if (cur->left != nullptr)
            {
                que.push(cur->left);
            }
            if (cur->right != nullptr)
            {
                que.push(cur->right);
            }
        }
    }
    return res;
}

void test4levelOrder()
{
    vector<vector<int>> res;
    TreeNode *root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(2);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(5);
    // root->right->left = new TreeNode(4);
    root->right->right = new TreeNode(3);
    Solution ss;
    res = ss.levelOrder(root);
    ss.printTree(root);
    for (auto r : res)
    {
        cout << "[ ";
        for (auto l : r)
        {
            cout << l << ' ';
        }
        cout << " ] ";
    }
    cout << " ]" << endl;
}

TreeNode *Solution::sortedArrayToBST(vector<int> &nums)
{
    TreeNode *root;

    function<TreeNode *(vector<int> &)> recur = [&](vector<int> &nums) -> TreeNode *
    {
        if (nums.size() < 1)
        {
            return nullptr;
        }
        if (nums.size() == 1)
        {
            return new TreeNode(nums[0]);
        }
        // 注意数组越界检查
        int mid = (nums.size() - 1) / 2;
        vector<int> left_nums(nums.begin(), nums.begin() + mid);
        vector<int> right_nums(nums.begin() + mid + 1, nums.end());
        TreeNode *r = new TreeNode(nums[mid]);
        r->left = recur(left_nums);
        r->right = recur(right_nums);
        return r;
    };
    root = recur(nums);
    return root;
}

TreeNode *Solution::sortedArrayToBST_1(vector<int> &nums)
{
    function<TreeNode *(vector<int> &, int, int)> recur = [&](vector<int> &nums, int left, int right) -> TreeNode *
    {
        if (left > right)
        {
            return nullptr;
        }
        // 注意数组越界检查
        int mid = (left + right) / 2;
        TreeNode *r = new TreeNode(nums[mid]);
        r->left = recur(nums, left, mid - 1);
        r->right = recur(nums, mid + 1, right);
        return r;
    };
    return recur(nums, 0, nums.size() - 1);
}

void test4sortedArrayToBST()
{
    vector<int> Arrs = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    Solution ss;
    TreeNode *res = ss.sortedArrayToBST_1(Arrs);
    ss.printTree(res);
}

// 35.找到返回下标，否则插入
int Solution::searchInsert(vector<int> &nums, int target)
{
    if (nums.size() <= 0)
    {
        return nums.size();
    }
    int less_mark = nums.size();
    for (int i = 0; i < nums.size(); ++i)
    {
        if (target == nums[i])
        {
            return i;
        }
        // 第一个比target大的，break
        if (target < nums[i])
        {
            less_mark = i;
            break;
        }
    }
    if (less_mark < nums.size())
    {
        return less_mark;
    }
    return nums.size();
}

// 35.使用二分查找
int Solution::searchInsert_1(vector<int> &nums, int target)
{
    int l = 0;
    int r = nums.size() - 1;
    int ans = nums.size();
    while (l <= r)
    {
        int mid = ((r - l) >> 1) + l;
        if (target <= nums[mid])
        {
            ans = mid;
            r = mid - 1;
        }
        else
        {
            l = mid + 1;
        }
    }
    return ans;
}

void test4searchInsert()
{
    vector<int> nums = {2, 4, 5, 9, 10};
    Solution ss;
    int res = ss.searchInsert_1(nums, 41);
    cout << res << endl;
}

bool Solution::isValid(std::string s)
{
    if (s.empty() || s.size() <= 1)
    {
        return false;
    }
    // vector<char> schemes =  {'(', ')', '{', '}', '[', ']'};
    unordered_map<char, int> ch_l = {
        {'(', 0},
        {'{', 1},
        {'[', 2},

    };
    unordered_map<char, int> ch_r = {
        {')', 0},
        {'}', 1},
        {']', 2}};
    stack<char> buff;

    for (int i = 0; i < s.size(); i++)
    {
        // 如果当前是左，则入栈，如果是右则与栈顶元素比较
        if (ch_l.count(s[i]))
        {
            buff.push(s[i]);
        }
        else
        { // 当前为右
            if (buff.size() == 0)
            {
                return false;
            }
            if (ch_l[buff.top()] != ch_r[s[i]])
            {
                return false;
            }
            buff.pop();
        }
    }
    return buff.size() == 0 ? true : false;
}

void test4isValid()
{
    string case1 = "[({(())}[()])]";
    Solution ss;
    bool res = ss.isValid(case1);
    cout << (res == 0 ? "false" : "true") << endl;
}

// 121.股票最大
int Solution::maxProfit(std::vector<int> &prices)
{
    // 记录历史最低
    int min_price = INT32_MAX;
    int max_profit = 0;
    for (auto price : prices)
    {
        max_profit = max(max_profit, price - min_price);
        min_price = min(min_price, price);
    }
    return max_profit;
}

void test4maxProfit()
{
    vector<int> prices = {3, 2, 6, 5, 0, 3};

    Solution ss;
    int res = ss.maxProfit(prices);
    cout << res << endl;
}

// 70.爬楼梯
int Solution::climbStairs(int n)
{
    // 递归
    function<int(int)> climb = [&](int m) -> int
    {
        if (m == 1)
        {
            return 1;
        }
        if (m == 2)
        {
            return 2;
        }
        return climb(m - 1) + climb(m - 2);
    };

    return climb(n);
}

int Solution::climbStairs_1(int n)
{
    vector<int> memo(n + 1);
    // 记忆递归，避免重复计算
    function<int(int, vector<int> &)> climb = [&](int m, vector<int> &memo) -> int
    {
        if (memo[m] > 0)
        {
            return memo[m];
        }
        if (m == 1)
        {
            memo[m] = 1;
        }
        else if (m == 2)
        {
            memo[m] = 2;
        }
        else
        {
            memo[m] = climb(m - 1, memo) + climb(m - 2, memo);
        }
        return memo[m];
    };

    return climb(n, memo);
}

int Solution::climbStairs_2(int n)
{
    // dp
    if (n <= 2)
    {
        return n;
    }
    vector<int> dp(n);
    dp[0] = 1;
    dp[1] = 2;
    for (int i = 2; i < n; ++i)
    {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp.back();
}

void test4climbStairs()
{
    int n = 6;
    Solution ss;
    int res0 = ss.climbStairs(n);
    int res1 = ss.climbStairs_1(n);
    int res2 = ss.climbStairs_2(n);

    cout << res0 << endl;
    cout << res1 << endl;
    cout << res2 << endl;
}

vector<vector<int>> Solution::generate(int numRows)
{
    // 初始值
    vector<vector<int>> triangle(numRows);
    if (numRows == 1)
    {
        triangle[0] = {1};
        return triangle;
    }
    if (numRows == 2)
    {
        triangle[1] = {1, 2};
        return triangle;
    }
    // 大于3// 从第3行开始，【0，【do this】，numsRows-1】
    triangle[0] = vector<int>(1, 1);
    triangle[1] = vector<int>(2, 1);

    for (int row = 2; row < numRows; row++)
    {
        vector<int> rowVec(row + 1); // 申请3个空间
        rowVec[0] = 1;
        rowVec[row] = 1;
        for (int col = 1; col < row; col++)
        { // 对中间操作[0,1,2]
            rowVec[col] = triangle[row - 1][col - 1] + triangle[row - 1][col];
        }
        triangle[row] = rowVec;
    }
    return triangle;
}

void test4generate()
{
    int numRows = 15;
    Solution ss;
    vector<vector<int>> res = ss.generate(numRows);

    ss.printVector2D(res);
}

vector<vector<string>> Solution::groupAnagrams(vector<string> &strs)
{
    vector<vector<string>> ans;
    if (strs.empty())
    {
        ans.push_back({""});
        return ans;
    }

    unordered_map<string, vector<string>> strs_hash_map;

    for (auto &s : strs)
    { // 不要改变原数据
        string ss = s;
        sort(ss.begin(), ss.end());        // key
        strs_hash_map[ss].emplace_back(s); // 利用emplce,优化1次查找机会
    }
    for (auto it = strs_hash_map.begin(); it != strs_hash_map.end(); it++)
    {
        ans.emplace_back(it->second);
    }

    return ans;
}

void test4groupAnagrams()
{
    vector<string> strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
    Solution ss;
    vector<vector<string>> res = ss.groupAnagrams(strs);
    ss.printVector2D(res);
}

// 128.最长连续序列长度
int Solution::longestConsecutive(std::vector<int> &nums)
{
    if (nums.size() <= 1)
    {
        return nums.size();
    }
    int res = 0;
    // 去重
    unordered_set<int> seq;
    for (auto num : nums)
    {
        seq.insert(num);
    }
    // 找最长，if nums[i] - 1 存在，则留下该点，否则跳过
    for (auto it = seq.begin(); it != seq.end(); it++)
    {
        if (seq.find((*it) - 1) == seq.end())
        { // not found
            // 当前为开头，寻找后面的
            int tmp_long = 1;
            for (auto after_it = (*it) + 1; seq.find(after_it) != seq.end(); after_it++, tmp_long++){}
            res = max(res, tmp_long); // 记录最长
        }
    }
    return res;
}

void test4longestConsecutive()
{
    vector<int> nums = {0, 3, 7, 2, 5, 8, 4, 6, 0, 1};
    Solution ss;
    int res = ss.longestConsecutive(nums);
    cout << res << endl;
}

//================END===================//

void myTest()
{
    // int a = 1;
    // int *pa = &a;
    // cout << *pa << endl;
    // int *pb, *pc = pa;
    // cout << *pb << endl;
    // cout << *pc << endl;

    string ss = "hello";
    string cc = ss;
    sort(ss.begin(), ss.end());
    cout << ss << endl;
    cout << cc << endl;
}