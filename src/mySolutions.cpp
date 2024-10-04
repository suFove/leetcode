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

void Solution::printVectorString(const vector<std::string> &vec)
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
        printVectorString(r);
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
    // 找最长，if nums[i] - 1 不存在，则该点为开头，向后查找，否则跳过
    for (auto it = seq.begin(); it != seq.end(); it++)
    {
        if (seq.find((*it) - 1) == seq.end())
        { // not found
            // 当前为开头，寻找后面的
            int tmp_long = 1;
            for (auto after_it = (*it) + 1; seq.find(after_it) != seq.end(); after_it++, tmp_long++)
            {
            }
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

// 11.盛水最多的容器
int Solution::maxArea(std::vector<int> &height)
{
    if (height.size() < 2)
        return 0;

    // main idea: max(2边x长), 使用双指针；
    int l = 0;
    int r = height.size() - 1;
    int res = 0;
    // 左右同时，寻找比当前值更高的，
    while (l < r)
    {
        // 记录最大
        res = max(res, (r - l) * min(height[l], height[r])); // 容量
        // 移动
        if (height[l] < height[r])
            l++;
        else
            r--;
    }

    return res;
}

void test4maxArea()
{
    vector<int> height = {1, 8, 6, 2, 5, 4, 8, 3, 7};
    Solution ss;
    int res = ss.maxArea(height);
    cout << res << endl;
}

vector<vector<int>> Solution::threeSum(vector<int> &nums)
{
    sort(nums.begin(), nums.end()); // 先排序，去重
    vector<vector<int>> ans;

    for (int i = 0; i < nums.size(); i++)
    {
        int j = i + 1, k = nums.size() - 1;

        while (j < k)
        {
            int sum = nums[i] + nums[j] + nums[k];

            if (sum == 0)
            {
                ans.push_back({nums[i], nums[j], nums[k]});
                j++;
                k--;

                // skip j and k
                while (j < k && nums[j] == nums[j - 1])
                    j++;
                while (j < k && nums[k] == nums[k + 1])
                    k--;
            }
            else if (sum < 0)
            {
                j++;
            }
            else
            {
                k--;
            }
        }

        // Skip i
        while (i + 1 < nums.size() && nums[i] == nums[i + 1])
            i++;
    }

    return ans;
}

void test4threeSum()
{
    vector<int> nums = {-1, 0, 1, 2, -1, -4};
    Solution ss;
    vector<vector<int>> res = ss.threeSum(nums);
    ss.printVector2D(res);
}

// 42.接雨水，求最大雨水
int Solution::trap(vector<int> &height)
{
    if (height.size() <= 2)
        return 0;
    int ans = 0;
    // 两端比中间高，
    // 初始化2个数组，保存从左（右）映射的最大值
    vector<int> leftMax(height.size()), rightMax(height.size());
    // 从左向右
    leftMax[0] = height[0];
    for (int i = 1; i < height.size(); i++)
    {
        leftMax[i] = max(leftMax[i - 1], height[i]);
    }
    // 从右向左
    rightMax[height.size() - 1] = height[height.size() - 1];
    for (int i = height.size() - 2; i >= 0; i--)
    {
        rightMax[i] = max(rightMax[i + 1], height[i]);
    }

    // 取这两个数组最小的元素
    for (int i = 0; i < height.size(); i++)
    {
        ans += min(leftMax[i], rightMax[i]) - height[i];
    }
    return ans;
}

int Solution::trap_1(vector<int> &height)
{
    if (height.size() <= 2)
        return 0;
    int ans = 0;
    // 与上个方法相近，考虑是否可以优化空间
    int l = 0;
    int r = height.size() - 1;
    int leftMax = 0, rightMax = 0;
    while (l < r)
    {
        leftMax = max(leftMax, height[l]);
        rightMax = max(rightMax, height[r]);
        // 从左边计算接水
        if (height[l] < height[r])
        {
            ans += leftMax - height[l++];
        }
        else
        {
            ans += rightMax - height[r--];
        }
    }
    return ans;
}

int Solution::trap_2(vector<int> &height)
{
    if (height.size() <= 2)
        return 0;
    int ans = 0;
    // 维护一个stack, 其栈顶元素是一侧的最大值（左边柱子），
    stack<int> buff_stcak; // 装载下标
    for (int i = 0; i < height.size(); i++)
    {
        while (!buff_stcak.empty() && height[buff_stcak.top()] < height[i])
        {                               // 保持左低右高 or 左右相等
            int top = buff_stcak.top(); // 坑底高度
            buff_stcak.pop();           // 弹出坑底
            if (buff_stcak.empty())
                break;
            // 计算:下标差*高度
            int left = buff_stcak.top(); // 左柱高度
            int index_diff = i - left - 1;
            int h = min(height[i], height[left]) - height[top];
            ans += index_diff * h;
        }
        buff_stcak.push(i);
    }
    return ans;
}

void test4trap()
{
    vector<int> height = {4, 2, 0, 3, 2, 5};
    Solution ss;
    int res = ss.trap_2(height);
    cout << res << endl;
}

// 3.最长无重复字串长度
int Solution::lengthOfLongestSubstring(string s)
{
    if (s.length() <= 1)
    {
        return s.length();
    }
    // 需要一个容器来承载无重复字串 and index
    unordered_map<char, int> smap;
    int ans = 0;
    for (int win_l = 0, win_r = 0; win_r < s.length(); ++win_r)
    {
        if (smap.find(s[win_r]) != smap.end())
        { // found
            // 找到不等于右边的第一个数,获得其下标
            while (win_l <= smap[s[win_r]])
            {
                smap.erase(s[win_l]);
                ++win_l;
            }
        }
        smap[s[win_r]] = win_r;
        ans = max(ans, (int)smap.size());
    }
    return ans;
}

void test4lengthOfLongestSubstring()
{
    string s = "aabaab!bb";
    Solution ss;
    int res = ss.lengthOfLongestSubstring(s);
    cout << res << endl;
}

// 438. 找到字符串中所有字母异位词
vector<int> Solution::findAnagrams(string s, string p)
{
    if (s.length() < p.length() || s.length() <= 0 || p.length() <= 0)
    {
        return {};
    }
    // s中匹配p<所有顺序>的字串起始下标
    //  window [] match {word: count}
    int sLen = s.size(), pLen = p.size();

    if (sLen < pLen)
    {
        return vector<int>();
    }

    vector<int> ans;
    vector<int> sCount(26);
    vector<int> pCount(26);
    // init
    for (int i = 0; i < pLen; ++i)
    {
        ++sCount[s[i] - 'a'];
        ++pCount[p[i] - 'a'];
    }
    // vector 与 vector 可以判定
    if (sCount == pCount)
    {
        ans.emplace_back(0);
    }
    // step = 1
    for (int i = 0; i < sLen - pLen; ++i)
    {
        --sCount[s[i] - 'a'];        // 清空上个窗口
        ++sCount[s[i + pLen] - 'a']; // update next window

        if (sCount == pCount)
        {
            ans.emplace_back(i + 1);
        }
    }
    return ans;
}

// 用差值来更新数值，不好理解
/**
如果 counter[s[i] - 'a'] == 1：这意味着在当前窗口中，某个字符 s[i] 的数量正好与字符串 p 中相同。由于我们正在从一个较大的窗口转移到较小的窗口（通过移除前一个字符），这意味着这个特定的字符现在不再是多余的。因此，它使得差异计数器 differ-- 减少1。
如果 counter[s[i] - 'a'] == 0：意味着在当前窗口中，某个字符的数量原本与字符串 p 中相同（即它是匹配的）。但随着窗口移动到下一个字符（即开始下一轮循环），该位置不再有这个字符。因此，差异计数器 differ++ 增加1。
如果 counter[s[i + p.length()] - 'a'] == -1：这表示在下一个循环中（即将成为窗口的一部分的字符），它原本不是窗口内的，但现在由于移动到下一次循环它将成为一个额外的字符。因此，差异计数器 differ-- 减少1。
如果 counter[s[i + p.length()] - 'a'] == 0：这意味着在当前窗口中，有一个额外的字符（即原本不在窗口内的），但是当窗口向前移动时，这个位置正好填补了之前移除的那个字符的位置。因此，差异计数器 differ++ 增加1
 * 
 */
vector<int> Solution::findAnagrams_1(string s, string p)
{
    if (s.length() < p.length())
        return {};

    vector<int> ans;
    vector<int> counter(26);
    // 不再使用2个数组分别记录，直接在更新时记录二者插值，判断是否为0，即相同
    // 第一个窗口init start
    for (int i = 0; i < p.length(); ++i)
    {
        --counter[s[i] - 'a']; 
        ++counter[p[i] - 'a'];
    }

    int differ = 0;
    for (int i = 0; i < p.length(); ++i)
    {
        if (counter[i] != 0)
        {
            ++differ;
        }
    }
    if (differ == 0)
    {
        ans.emplace_back(0);
    }
    // end first window

    for (int i = 0; i < s.length() - p.length(); ++i)
    {
        //  erase window left
        if (counter[s[i] - 'a'] == 1) //当前多余，erase之后，differ需要减少该位置的差异
        { 
            --differ;
        }
        else if (counter[s[i] - 'a'] == 0) //当前匹配，erase之后，就不匹配了
        { 
            ++differ;
        }
        --counter[s[i] - 'a'];
        // end erase

        // add window right
        if (counter[s[i + p.length()] - 'a'] == -1) // 当前不匹配，add之后需要减少当前位置差异
        { 
            --differ;
        }
        else if (counter[s[i + p.length()] - 'a'] == 0) // 当前匹配，add之后需要增加当前位置差异
        { 
            ++differ;
        }
        ++counter[s[i + p.length()] - 'a'];
        // end add
        if (differ == 0)
        {
            ans.emplace_back(i + 1);
        }
    }
    return ans;
}

void test4findAnagrams()
{
    string s = "cbaebabacd";
    string p = "abc";
    Solution ss;
    vector<int> res = ss.findAnagrams_1(s, p);
    ss.printVector(res);
}

//560. 连续子数组的个数： condition: sum = target
/**
 * main idea: 寻找一个{seq | sum(seq) = target}
 * 变化 win_l and win_r; 或者使用队列，但是队列对seq sum没有优化，实际还是下标操作
 * main loop: 
 *      continue condition: sum(seq) < target
 *      end condition: seq[i] == target or sum(seq) == target, change window, 弹出队列头部元素
 *      optimize condition: if x > traget exsit in seq, change window
 * note: 以上只考虑了正数情况
 */
int Solution::subarraySum(vector<int>& nums, int k){
    // O(n^2)
    int ans = 0;
    for(int cur = 0; cur < nums.size(); ++cur){
        int sum_nums = 0;
        for(int tmp_end = cur;  tmp_end >= 0; --tmp_end){//实时统计sum
            //update
            sum_nums += nums[tmp_end];
            if(sum_nums == k){
                ++ans; 
            }
        }   
    }
    return ans;
}

//560. 考虑优化:上面方法加入一个元素，就要重新求已经不满足条件的nums[x:y]的和
/**
 * 总结：用前缀和sum(nums[:pre])来作为k，检查所有前缀和出现的次数
 * 条件：nums[cur] + sum(nums[:i-1]) == target ; 其中 sum(nums[:i-1]) 保存为hash key,
 * 只需在 hash map 找到 nums[cur] - target == sum(nums[:cur-1]) 是否存在 
 * 
 * 初始插入{0，1}，如果满足条件，即pre-k == 0, 则更新ans += {k,v}
 * 例如 target = 3
 * 第二个元素应该是{3，1}，下次 pre-k == 3， 则说明 当前满足条件：cur + pre == target
 * 
 */

int Solution::subarraySum_1(vector<int>& nums, int k){
    int ans = 0;
    unordered_map<int, int> mp;
    mp[0] = 1;
    int pre = 0;
    for (auto& x:nums) {
        pre += x;
        if (mp.find(pre - k) != mp.end()) {
            ans += mp[pre - k];
        }
        mp[pre]++;
    }
    return ans;
}


void test4subarraySum(){
    vector<int> nums = {-1,-1,1};
    int target = 0;
    Solution ss;
    int res  = ss.subarraySum_1(nums, target);
    cout << res << endl;
}



/**
 * 
239. 滑动窗口最大值
like max pooling 
 result: 超时 O(n^win_size) ~= O(n^2)
 考虑优化 window 使用优先队列 O(nlgn)
 */
vector<int> Solution::maxSlidingWindow(vector<int>& nums, int k)
{
    vector<int> ans;
    priority_queue<int> q;
    // 1 <= k <= nums.length
    if(k == nums.size()){
        ans.emplace_back(*max_element(nums.begin(), nums.end()));
        return ans;
    }
    for(int i = 0; i <= nums.size() - k; ++i){
        for(int win_l = i; win_l < i + k; ++win_l){
            // max = nums[win_l] > max ? nums[win_l] : max;
            q.push(nums[win_l]);
        }
        ans.emplace_back(q.top());
        q = {}; // clear
    }
    return ans;
}



/**
 * 
239. 滑动窗口最大值
like max pooling 
 考虑优化:使用双端队列，2端插入删除为O(1)
 note：deque 内部存储不是连续的，需要保留原数组的index
    example: window_size = 3;
    init [1, 2]
    next elem = 3 -> [] -> [3]
    
 */
vector<int> Solution::maxSlidingWindow_1(vector<int>& nums, int k){
    if (k == 1 || k > nums.size()) return nums;

    deque<int> q;// 储存index
    
    // init first window; 
    for(int i = 0; i < k; ++i){
        while( !q.empty() && nums[i] >= nums[q.back()]){
            q.pop_back();
        }//弹出所有比当前小的元素
        //加入最大元素
        q.emplace_back(i);
    }
    vector<int> ans = {nums[q.front()]};

    // next windows
    for(int i = k; i < nums.size(); ++i){
        // filter
        while(!q.empty() && nums[i] >= nums[q.back()]){
            q.pop_back();
        }
        q.emplace_back(i);
        // 维护窗口大小, [0, 1, 2, 3](i = 3) -> [1, 2, 3]
        while(q.front() <= i-k){
            q.pop_front();
        }
        ans.emplace_back(nums[q.front()]);
    }

    return ans;

}


void test4maxSlidingWindow(){
    vector<int> nums = {1,3,-1,-3,5,3,6,7};
    int win_size = 3;
    Solution ss;
    vector<int> res = ss.maxSlidingWindow_1(nums, win_size);
    ss.printVector(nums);
    ss.printVector(res);
}


//76. s涵盖所有t的字串
/**
 * main idea: 
 *      t -> {word : count}
 *      s -> [pop, push_back], check window word count
 *      不满足条件win_r++，直到满足条件
 *      满足条件 win_l++，直到不满足条件
 */
string Solution::minWindow(string s, string t){
    
    int ans_left = -1;
    int min_len = INT_MAX;
    unordered_map<char, int> mp;
    unordered_map<char, int> mq;
    queue<int> q; // cache index


    for(auto c : t){
        ++mp[c];
    }

    //匹配函数：t 中 word count > window 中 word count
    function<bool()> check =[&]()-> bool{
        for(auto & scheme : mp){
            if(scheme.second > mq[scheme.first])
                return false;
        }
        return true;
    };

    // loop
    for(int win_l, win_r = 0; win_r < s.length(); win_r++) {
        //遍历元素 更新mq
        ++mq[s[win_r]];

        //检查当前元素是否存在在scheme中
        if(mp.find(s[win_r]) != mp.end()){
            //对于每一次有效的更新，检查当前窗口中的子串是否是一个合格的子串
            while(check() && win_l <= win_r){
                //当前合格，判断是否最小，并记录
                if(min_len > win_r - win_l + 1){
                    ans_left = win_l;
                    min_len = win_r - win_l + 1;
                }
                // 尝试缩短窗口
                --mq[s[win_l]];
                ++win_l;
            }
        }
    }

    if(ans_left == -1) 
        return "";
    else
        return s.substr(ans_left, min_len);
}

void test4minWindow(){
    string s = "ADOBECODEBANC";
    string t = "ABC";
    Solution ss;
    string res = ss.minWindow(s, t);
    cout << res << endl;
}

//53. 找到最大和的子数组
/**
 * main idea:
 *      window left and right: 
 *      [-1] -> [-1, 2] -> [2]
 *      win_r 加入产生最大值:
 *      while(试图删除 win_l | win_r <= win_r)
 *      max seq[i] = max seq[i-1] 
 * 只能完成O(n^2) 方法，因为要回看max
 * 
 * 尝试修改：
 *      main idea : 若前一个元素 > 0; 则加到当前元素上
 *      动态规划状态转移方程
 *      f(i) = max{ f(i−1) + nums[i] , nums[i] }
 * 
 * 不难给出一个时间复杂度 O(n)、空间复杂度 O(n) 的实现，
 * 即用一个 f 数组来保存 f(i) 的值，用一个循环求出所有 f(i)。考虑到 f(i) 只和 f(i−1) 相关，
 * 于是我们可以只用一个变量 pre 来维护对于当前 f(i) 的 f(i−1) 的值是多少，从而让空间复杂度降低到 O(1)，
 * 这有点类似「滚动数组」的思想。
 */
int Solution::maxSubArray(vector<int>& nums){
    vector<int> dp = nums;
    int ans = 0;
    //若前一个元素 > 0， 则加上当前元素
    for(int i = 1; i < nums.size(); ++i){
        if(dp[i-1] > 0){
            dp[i] = dp[i-1] + nums[i];
        }
    }
    ans = *max_element(dp.begin(), dp.end());
    return ans;
}

/**
 * 贪心：如果 sum[0:i] < 0, 则丢弃前面的和
 *  
 */
int Solution::maxSubArray_1(vector<int>& nums){
    int max_sum = nums[0];
    int cur_sum = 0;
    for(auto & x : nums){
        cur_sum = max(cur_sum + x, x); // 当前和 = 之前和 + 当前值 | 当前值（去掉之前和）
        max_sum = max(max_sum , cur_sum); //记录最大值
    }
    return max_sum;
}

void test4maxSubArray(){
    vector<int> nums = {-2,1,-3,4,-1,2,1,-5,4};
    Solution ss;
    int res = ss.maxSubArray(nums);
    // int res = ss.maxSubArray_1(nums);
    cout << res << endl;
}

// 56. 每个内部元素都是长度为2的数组 [ [1,4] [1,5] ] -> [[4,5]] return 一个不重叠的区域
vector<vector<int>> Solution::merge(vector<vector<int>>& intervals){
    vector<vector<int>> ans;
    for(auto &x : intervals)
        if(x[0] > x[1])
            swap(x[0], x[1]);
    sort(intervals.begin(), intervals.end());
    printVector2D(intervals);

    ans.emplace_back(intervals[0]);
    for(int i = 1; i < intervals.size(); ++i){
        //从第2个元素开始，找结束标记， 重叠条件
        if( (ans.back()[0] <= intervals[i][0] && intervals[i][0] <= ans.back()[1]) || (intervals[i][0] <= ans.back()[1] && ans.back()[1] <= intervals[i][1]))
        {   
            int first = min(intervals[i][0], ans.back()[0]);
            int second = max(intervals[i][1], ans.back()[1]);
            vector<int> tmp = {first, second};
            //当前与前一个发生重叠，考虑加入ans
            ans.back() = tmp;
        }else{//不重叠 直接加入
            ans.emplace_back(intervals[i]);
        }     
    }

    return ans;
}


void test4merge(){
    vector<vector<int>> intervals = {{1,3},{2,6},{8,10},{15,18}};
    // vector<vector<int>> intervals = {{1,4},{0,4}};
    // vector<vector<int>> intervals = {{1,4},{0,2}, {3,5}};
    Solution ss;
    vector<vector<int>> res = ss.merge(intervals);
    ss.printVector2D(res);
}

//189. 轮转数组，类似于马戏团传送带，向右移动次数k
void Solution::rotate(vector<int>& nums, int k){
    //length = 1
    if(nums.size() == 1) return;
    //最容易想到是新建数组，定位起始和终止下标
    vector<int> cy_nums;
    int start_idx = nums.size() - (k % nums.size());
    int end_idx = start_idx-1;
    while(start_idx < nums.size()){
        cy_nums.emplace_back(nums[start_idx]);
        ++start_idx;
    }
    for(int i = 0; i <= end_idx; ++i){
        cy_nums.emplace_back(nums[i]);
    }
    nums = cy_nums;
}

// 189.
void Solution::rotate_1(vector<int>& nums, int k){
    //length = 1
    if(nums.size() == 1) return;
    // 利用反转字串
    vector<int> cy_nums;
    int start_idx = nums.size() - (k % nums.size());
    // 反转尾部， 反转头部， 反转整体
    reverse(nums.begin() + start_idx, nums.end());
    reverse(nums.begin(), nums.begin() + start_idx);
    reverse(nums.begin(), nums.end());
}

void test4rotate(){
    vector<int> nums = {1,2,3,4,5,6,7};
    int k = 10;
    Solution ss;
    ss.rotate_1(nums, k);
    ss.printVector(nums);   
}

//238. 除了自身以外数组的乘积
/**
 * 实际是找子数组的乘积，想办法把子数组保存利用
 * [1,2,3,4] -> [ [2,3,4], [1,3,4], [1,2,4], [1,2,3] ]
 *
 * [2,3,4,1]
 * [3,4,1,2]
 * [4,1,2,3]
 * 
 * 求所有列的乘积时间复杂度是 O(N*(N-1)) ~ O(N^2)
 * 上述方案 pass
 * 
 * O(n)解决方案：
 *  将当前idx看作分界线，只需要考虑 左子数组 * 右子数组 即可
 */
vector<int> Solution::productExceptSelf(vector<int>& nums){
    
    //curr = left[last] * nums[last]
    vector<int> left_muti(nums.size());// 从左向右 所有乘积
    vector<int> right_muti(nums.size());// 从右向左 所有乘积
    vector<int> ans(nums.size());
    // 初始化为 1
    left_muti[0] = right_muti[nums.size()-1] = 1;

    for(int i = 1; i < nums.size(); ++i){
        left_muti[i] = left_muti[i-1] * nums[i-1];
    }

    for(int j = nums.size()-2; j >=0; --j){
        right_muti[j] = right_muti[j+1] * nums[j+1];
    }

    for(int k = 0; k < nums.size(); ++k){
        ans[k] = left_muti[k] * right_muti[k];
    }

    return ans;
}


void test4productExceptSelf(){
    vector<int> nums = {1,2,3,4};
    Solution ss;
    vector<int> res =  ss.productExceptSelf(nums);
    ss.printVector(res);

}

//41. 缺失的第一个正数
/**
 * 数组未排序，不连续
 * 最简单的 hash map, 全部放入，如果长度==数组长度，则返回length+1
 * 技巧：需要处理负数
 *      1. 负数转为正数，数值为 length+1 {目的：如果全是负数，则会返回 length + 1, 符合题目假设}
 *      2. 原数值小于 length , 则对应 “排序的位置（数值所旨位置）” 转为负数 {目的：转为负数，以便查找正数的连续性}
 *      3. 返回第一个 正数的下标+1
 */
int Solution::firstMissingPositive(vector<int>& nums){
    int n = nums.size();
    for (int& num: nums) {
        if (num <= 0) {
            num = n + 1;
        }
    }
    for (int i = 0; i < n; ++i) {
        int num = abs(nums[i]);
        if (num <= n) {
            nums[num - 1] = -abs(nums[num - 1]);
        }
    }
    // 返回第一个正数
    for(int i = 0; i < nums.size(); ++i){
        if(nums[i] > 0){
            return i+1;
           
        }         
    }
    return nums.size()+1;
}


// 方法 2：
/**
 * 采用置换手段
 * 如果位置全部正确，置换完应有【1，2，3，4....n】，即 数组的第 x−1 个元素为 x
 * 考虑数组恢复方法：
 *      如果当前元素 数值在【1，n】范围，则检查 nums【value-1】位置已经存在value，不存在则置换
 */
int Solution::firstMissingPositive_1(vector<int>& nums){
    int length = nums.size();
    for(int i = 0; i < length; ++i){
        while(nums[i] > 0 && nums[i] <= length && nums[nums[i] - 1] != nums[i]){
            swap(nums[nums[i] - 1], nums[i]);
        }
    }
    for(int i = 0; i < length; ++i){
        if(nums[i] != i+1){
            return i+1;
        }
    }
    return length+1;
}

void test4firstMissingPositive(){
    vector<int> nums = {3,4,-1,1};
    Solution ss;
    int res1 = ss.firstMissingPositive(nums);
    int res2 = ss.firstMissingPositive_1(nums);
    cout << res1 << endl;
    cout << res2 << endl;
}


//73. 矩阵置零
/**
 * 使用额外空间来记录应该标记的地点
 * 可以使用 m x n 的大空间
 * 考虑优化： 使用2个数组，长度分别为m和n，也可以记录
 */
void Solution::setZeroes(vector<vector<int>>& matrix){
    int m = matrix.size();
    int n = matrix[0].size();
    // 额外空间记录
    vector<bool> row_tag(m);
    vector<bool> col_tag(n);

    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            if(matrix[i][j] == 0){
                row_tag[i] = true;
                col_tag[j] = true;
            }
        }
    }

    for(int i = 0; i < m; ++i){
        if(row_tag[i]){
            for(auto & x : matrix[i]){
                x = 0;
            } 
        }
        
    }
    for(int j = 0; j < n; ++j){
        if(col_tag[j]){
            for(auto & x : matrix){
                x[j] = 0;
            } 
        }
    }
}


void test4setZeroes(){
    vector<vector<int>> matrix = {  {1,1,1},
                                    {1,0,1},
                                    {1,1,1} };
    Solution ss;
    cout << "The original:" << endl;
    ss.printVector2D(matrix);
    
   
    ss.setZeroes(matrix);
    cout << "Seted zero:" << endl;
    ss.printVector2D(matrix);

}

//334. 递增的三元子序列
bool Solution::increasingTriplet(std::vector<int>& nums){
    // 维护 min and max 2 个变量，表示一条折线的最低点和最高点
    // true 条件： 出现了比当前 max 更大的 元素；
    int start_min = nums[0];
    int end_max = INT_MAX;

    for(int i = 1; i < nums.size(); ++i){
        // cout << i << ' ';
        // 满足条件
        if(nums[i] > end_max) 
            return true;
        if(nums[i] > start_min) // 老折线头 + 新顶峰
            end_max = nums[i];
        else{// 新的折线头
            start_min = nums[i];
            // end_max = INT_MAX; // 不用添加重置max条件，保留之前折线的可用性
        }
    } 
    return false;
}

// 返回最大 递增 三元组 的值
int Solution::getIncreasingTripletValue(std::vector<int>& nums){
    //找最大三元组
    deque<int> ans;// 用于储存三元组的
    ans.emplace_back(nums[0]);
    // 记录最大值下标
    // int max_idx_start = 0;
    int max_value = INT_MIN;
    // 从 frist window + step 1 开始
    for(int i = 1; i < nums.size(); ++i){
        int cur_value = 0;// 记录当前递增三元组的值
        // 数组满3个，考虑弹出
        if(ans.size() == 3){
            cur_value = accumulate(ans.begin(), ans.end(), 0);
            if( max_value < cur_value){ // 更新
                // max_idx_start = i;
                max_value = cur_value;
            }
            ans.pop_front(); // 弹出前端
        }
        // 满足递增则入
        if(nums[i] > ans.back()){
            ans.push_back(nums[i]);
        }  
    }

    if(ans.size() == 3){
        int last_value = accumulate(ans.begin(), ans.end(), 0);
        return max_value > last_value ? max_value : last_value;
    }
    return max_value != INT_MIN ? max_value : 0;
}


void test4increasingTriplet(){
    vector<int> nums = {20,100,10,101,5,13};
    Solution ss;
    // int res = ss.getIncreasingTripletValue(nums);
    bool res = ss.increasingTriplet(nums);
    cout << res << endl;
}


// 54. 返回 mxn 矩阵的顺时针访问顺序
/**
 * 转换一下思维：缩小边界条件
 */
vector<int> Solution::spiralOrder(vector<vector<int>>& matrix){
    vector<int> ans;
    if(matrix.empty()) return ans; //若数组为空，直接返回答案
    // 定义上下左右边界, 注意是下标
    int top = 0;
    int bottom = matrix.size() - 1; 
    int left = 0;
    int right = matrix[0].size() - 1;
    // loop, 结束条件 上下重合，或者左右重合
    while(true){
         // 移动最上方，向右移动
        for(int i = left; i <= right; ++i)   ans.emplace_back(matrix[top][i]);
        if(++top > bottom)  break; // 缩减上方边界
         // 移动最右端，从上到下
        for(int i = top; i <= bottom; ++i)     ans.emplace_back(matrix[i][right]);
        if(--right < left)  break; // 缩减右方边界
         // 最顶端，从右向左
        for(int i = right; i >= left; --i)  ans.emplace_back(matrix[bottom][i]);
        if(--bottom < top)  break; // 缩减下方边界
         // 最左边，从下到上
        for(int i = bottom; i >= top; --i)  ans.emplace_back(matrix[i][left]);
        if(++left > right)  break; // 缩减左方边界
    }
    return ans;
}

void test4spiralOrder(){
    vector<vector<int>> matrix = {  {1,2,3},
                                    {4,5,6},
                                    {7,8,9} };
    Solution ss;
    cout << "The original:" << endl;
    ss.printVector2D(matrix);
    
   
    vector<int> res = ss.spiralOrder(matrix);
    cout << "Sprial sequence:" << endl;
    ss.printVector(res);
}

//48. 原地旋转二维矩阵  n × n
/**
 * 1.寻找位置映射关系： [i,j] -> [j, i]; 使用辅助数组
 * 2.原地方法：沿主对角线为对称轴，交换
 *            对每一行进行reverse，即可得到target数组
 * 
 */
void Solution::rotate(vector<vector<int>>& matrix){
    int n = matrix.size();
    // 对称交换
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < i; ++j){
            if(i == j) continue;
            swap(matrix[i][j], matrix[j][i]);
        }
    }
    // 反转每一行
    for(auto &line: matrix){
        reverse(line.begin(), line.end());
    }
}


void test4rotateMatrix(){
    vector<vector<int>> matrix = {  {1,2,3},
                                    {4,5,6},
                                    {7,8,9} };
    Solution ss;
    cout << "The original:" << endl;
    ss.printVector2D(matrix);
    
   
    ss.rotate(matrix);
    cout << "Rotated matrix:" << endl;
    ss.printVector2D(matrix);
}


//240. 搜索二维矩阵的target
/**
 * 1.暴力法
 */
bool Solution::searchMatrix(vector<vector<int>>& matrix, int target){
    for(auto &line : matrix){
        for(auto &ele : line){
            if(target == ele)
                return true;
        }
    }
    return false;
}

/**
 * 思考优化方法
 * 已知：
 *  m x n 矩阵
 *  每行的元素从左到右升序排列。每列的元素从上到下升序排列。
 *  使用二叉搜索确定行，再用二叉搜索确定列
 *  O(LogN+MLogN)
 */
bool Solution::searchMatrix_1(vector<vector<int>>& matrix, int target){
   

    // 二分查找 辅助函数
    function<int (vector<vector<int>>&, int, int, int)> halfSearch = [&](vector<vector<int>>& matrix, int row = -1, int col = -1, int target = 0) -> int{
        // 通过参数确定从行查找还是从列查找
        int m = matrix.size();
        int n = matrix[0].size();
        
        // 未规定行，从列开始查找
        if(row == -1){
            int top = 0;
            int bottom = m-1;
            int mid_line = 0;
            while(top <= bottom){
                mid_line = top + (bottom - top)/2;
                if(matrix[mid_line][col] > target){
                    bottom = mid_line -1;
                }else if( matrix[mid_line][col] < target){
                    top = mid_line + 1;
                }else{
                    return mid_line;
                }    
            }
            return mid_line; // 最近 大于target的哪一行
        }
        // 未规定列，从行还是查找
        if(col == -1){
            int left = 0;
            int right = n-1;
            int mid_col = 0;
            while(left <= right){
                mid_col = left + (right - left)/2;
                if(matrix[row][mid_col] > target){
                    right = mid_col-1;
                }else if( matrix[row][mid_col] < target){
                    left = mid_col + 1;
                }else{ // 找到target
                    return mid_col;
                }    
            }
            return -1;
        }   
        return -1;
    };

    int row_idx = halfSearch(matrix, -1, 0, target);
    if(row_idx == -1) return false;
    for(int i = row_idx; i >= 0; --i){ // 对所有满足条件的行进行遍历
        int col_idx = halfSearch(matrix, i, -1, target);
        if(col_idx >= 0) return true;
    }
    return false;
}

/**
 * 因为是有序的，考虑从结点路径改变。
 * 类似于二叉搜索树
 * 
 */
bool Solution::searchMatrix_2(vector<vector<int>>& matrix, int target){
    int m =matrix.size();
    int n =matrix[0].size();
    int row = 0;
    int col = n-1; // 从每行最后一个元素开始，便于跳行，或是锁定该行
    while(row < m && col >= 0){
        // 结束条件
        if(matrix[row][col] == target)
            return true;

        if(matrix[row][col] > target){
            --col;
        }else{ // curr < target -> 需要跳行了
            ++row;
        }
    }
    return false;
}


void test4searchMatrix(){
    // vector<vector<int>> matrix = {  {1,4,7,11,15},
    //                                 {2,5,8,12,19},
    //                                 {3,6,9,16,22},
    //                                 {10,13,14,17,24},
    //                                 {18,21,23,26,30}    };
    vector<vector<int>> matrix = { {5}, {6} };
    Solution ss;
    cout << "The original:" << endl;
    ss.printVector2D(matrix);
    
   
    bool res = ss.searchMatrix_2(matrix, 6);
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

    // string ss = "hello";
    // string cc = ss;
    // sort(ss.begin(), ss.end());
    // cout << ss << endl;
    // cout << cc << endl;
    // int a[][3] = {1,2,3,4,5,6};
    // int (*p)[3] = a;
    // cout << (*p)[1] << (*p)[2] << endl;

    // p++;

    // cout << (*p)[1] << (*p)[2] << endl;


    // int a[2][3] = {0,1,2,3,4,5};
    // int *p = &a[0][0];
    // cout << p[1*3 + 0] << endl;
    
    // a [] = {231312}
    // b [] = {没初始化}
    // c [] = {}
    // int a[] = {1,2,3,4,5,6};
    // int * p1 = (int *)(&a+1);
    // int * p2 = (int * )&a+1; 
    // cout << *(p1-0) << endl;
    // cout << *(p1-1) << endl;
    // cout << *p2 << endl;

    // enum string{
    //     x1, x2, x3=10, x4, x5
    // }x;
    // cout << x << endl;

    // char buffer[6] = {0};
    // char *s = "Hello World!";
    // for (int i = 0; i < sizeof(buffer) - 1; i++)
    // {
    //     buffer[i] = *(s + i);
    // }
    // for(auto x : buffer)
    //     cout << x << endl;


    // int a[10];

    // int * e[10];
    // int (* f)[10];
    // int (*a)(int);
    // int(*a[10])(int);

    vector<int> tmp;
    int i = 0;
    int number = 0;
    string line, word;
    getline(cin, line);
    //将记录绑定到刚刚读取的行，用到了sstream里面的对象
    istringstream record(line);
    while (record >> word)
    {
        tmp.push_back(atoi(word.c_str()));
    }
    for(auto x: tmp){
        cout << x << " ";
    }
}       