#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <vector>
#include <numeric>
#include <algorithm>
#include <exception>
#include <functional>



namespace kdt
{
	/** @brief k-d tree class.
	*/
	template <class PointT>
	class KDTree
	{
	public:
		/** @brief The constructors.
		*/
		KDTree() : root_(nullptr) {};
		KDTree(const std::vector<PointT>& points) : root_(nullptr) { build(points); }

		/** @brief The destructor.
		*/
		// ~KDTree() { clear(); }

		/** @brief Re-builds k-d tree.
		*/
		void build(const std::vector<PointT>& points)
		{
			// clear();

			points_ = points;

			std::vector<int> indices(points.size());
			std::iota(std::begin(indices), std::end(indices), 0);

			root_ = buildRecursive(indices.data(), (int)points.size(), 0);
		}



		// /** @brief Searches the nearest neighbor.
		// */
		int nnSearch(const PointT& query, double* minDist = nullptr) const
		{
			int guess;
			double _minDist = std::numeric_limits<double>::max();

			nnSearchRecursive(query, root_, &guess, &_minDist);

			if (minDist)
				*minDist = _minDist;

			return guess;
		}

		/** @brief Searches neighbors within radius.
		*/
		std::vector<int> radiusSearch(const PointT& query, double radius) const
		{
			std::vector<int> indices;
			radiusSearchRecursive(query, root_, indices, radius);
			return indices;
		}

	private:

		/** @brief k-d tree node.
		*/
		struct Node
		{
			int idx;       //!< index to the original point
			Node* next[2]; //!< pointers to the child nodes
			int axis;      //!< dimension's axis

			Node() : idx(-1), axis(-1) { next[0] = next[1] = nullptr; }
		};

		/** @brief k-d tree exception.
		*/
		class Exception : public std::exception { using std::exception::exception; };

		/** @brief Bounded priority queue.
		*/
		template <class T, class Compare = std::less<T>>
		class BoundedPriorityQueue
		{
		public:

			BoundedPriorityQueue() = delete;
			BoundedPriorityQueue(size_t bound) : bound_(bound) { elements_.reserve(bound + 1); };

			void push(const T& val)
			{
				auto it = std::find_if(std::begin(elements_), std::end(elements_),
					[&](const T& element){ return Compare()(val, element); });
				elements_.insert(it, val);

				if (elements_.size() > bound_)
					elements_.resize(bound_);
			}

			const T& back() const { return elements_.back(); };
			const T& operator[](size_t index) const { return elements_[index]; }
			size_t size() const { return elements_.size(); }

		private:
			size_t bound_;
			std::vector<T> elements_;
		};

		/** @brief Priority queue of <distance, index> pair.
		*/
		using KnnQueue = BoundedPriorityQueue<std::pair<double, int>>;

		/** @brief Builds k-d tree recursively.
		*/
		Node* buildRecursive(int* indices, int npoints, int depth)
		{
			if (npoints <= 0)
				return nullptr;

			const int axis = depth % PointT::DIM;
			const int mid = (npoints - 1) / 2;

			std::nth_element(indices, indices + mid, indices + npoints, [&](int lhs, int rhs)
			{
				return points_[lhs][axis] < points_[rhs][axis];
			});

			Node* node = new Node();
			node->idx = indices[mid];
			node->axis = axis;

			node->next[0] = buildRecursive(indices, mid, depth + 1);
			node->next[1] = buildRecursive(indices + mid + 1, npoints - mid - 1, depth + 1);

			return node;
		}


		static double distance(const PointT& p, const PointT& q)
		{
			double dist = 0;
			for (size_t i = 0; i < PointT::DIM; i++)
				dist += (p[i] - q[i]) * (p[i] - q[i]);
			return sqrt(dist);
		}

		/** @brief Searches the nearest neighbor recursively.
		*/
		void nnSearchRecursive(const PointT& query, const Node* node, int *guess, double *minDist) const
		{
			if (node == nullptr)
				return;

			const PointT& train = points_[node->idx];

			const double dist = distance(query, train);
			if (dist < *minDist)
			{
				*minDist = dist;
				*guess = node->idx;
			}

			const int axis = node->axis;
			const int dir = query[axis] < train[axis] ? 0 : 1;
			nnSearchRecursive(query, node->next[dir], guess, minDist);

			const double diff = fabs(query[axis] - train[axis]); //???axis???????????????
			if (diff < *minDist)
				nnSearchRecursive(query, node->next[!dir], guess, minDist);
		}

		/** @brief Searches neighbors within radius.
		*/
		void radiusSearchRecursive(const PointT& query, const Node* node, std::vector<int>& indices, double radius) const
		{
			if (node == nullptr)
				return;

			const PointT& train = points_[node->idx];

			const double dist = distance(query, train);
			if (dist < radius)
				indices.push_back(node->idx);

			const int axis = node->axis;
			const int dir = query[axis] < train[axis] ? 0 : 1;
			radiusSearchRecursive(query, node->next[dir], indices, radius);

			const double diff = fabs(query[axis] - train[axis]);
			if (diff < radius)
				radiusSearchRecursive(query, node->next[!dir], indices, radius);
		}

		Node* root_;                 //!< root node
		std::vector<PointT> points_; //!< points
	};
} // kdt



#endif // !__KDTREE_H__
