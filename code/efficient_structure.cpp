#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class MosesDataList {
public:
    MosesDataList() {
    }

    void add(const std::vector<std::tuple<std::string, int, int>> &sent) {
        list_.emplace_back(sent);
    }

    std::vector<std::tuple<std::string, int, int>> index(long i) const {
        return list_[i];
    }

    long size() {
        return list_.size();
    }

private:
    std::vector<std::vector<std::tuple<std::string, int, int> > > list_;
};

class MosesAlignmentList {
public:
    MosesAlignmentList() {
    }

    void add(const std::vector<std::tuple<int, int, int, int> > &sent) {
        list_.emplace_back(sent);
    }

    std::vector<std::tuple<int, int, int, int> > index(long i) const {
        return list_[i];
    }

    long size() {
        return list_.size();
    }

private:
    std::vector<std::vector<std::tuple<int, int, int, int> > > list_;
};

class MosesPhraseIndexList {
public:
    MosesPhraseIndexList() {
    }

    void add(const std::vector<std::tuple<int, int>> &sent) {
        list_.emplace_back(sent);
    }

    std::vector<std::tuple<int, int>> index(long i) const {
        return list_[i];
    }

    long size() {
        return list_.size();
    }

private:
    std::vector<std::vector<std::tuple<int, int> > > list_;
};

class PhraseDataList {
public:
    PhraseDataList() {
    }

    void add(const std::vector<std::tuple<long, int, int>> &sent) {
        list_.emplace_back(sent);
    }

    std::vector<std::tuple<long, int, int>> index(long i) const {
        return list_[i];
    }

    long size() {
        return list_.size();
    }

private:
    std::vector<std::vector<std::tuple<long, int, int> > > list_;
};

class EncoderDataList {
public:
    EncoderDataList() {
    }

    void add(const std::vector<long> &sent) {
        list_.emplace_back(sent);
    }

    std::vector<long> index(long i) const {
        return list_[i];
    }

    long size() {
        return list_.size();
    }

private:
    std::vector<std::vector<long> > list_;
};

class ChunkDataList {
public:
    ChunkDataList() {
    }

    void add(const std::vector<std::vector<long> > &it) {
        list_.emplace_back(it);
    }

    std::vector<std::vector<long>> index(long i) const {
        return list_[i];
    }

    long size() {
        return list_.size();
    }

private:
    std::vector<std::vector<std::vector<long> > > list_;
};



PYBIND11_MODULE(efficient_structure, m) {
    m.doc() = "Efficient Structure";

    py::class_<MosesDataList>(m, "MosesDataList")
        .def(py::init())
        .def("add", &MosesDataList::add, "Add a sent")
        .def("size", &MosesDataList::size, "Get size")
        .def("index", &MosesDataList::index, "Get the sent by index");

    py::class_<MosesAlignmentList>(m, "MosesAlignmentList")
        .def(py::init())
        .def("add", &MosesAlignmentList::add, "Add a sent")
        .def("size", &MosesAlignmentList::size, "Get size")
        .def("index", &MosesAlignmentList::index, "Get the sent by index");

    py::class_<MosesPhraseIndexList>(m, "MosesPhraseIndexList")
        .def(py::init())
        .def("add", &MosesPhraseIndexList::add, "Add a sent")
        .def("size", &MosesPhraseIndexList::size, "Get size")
        .def("index", &MosesPhraseIndexList::index, "Get the sent by index");

    py::class_<PhraseDataList>(m, "PhraseDataList")
        .def(py::init())
        .def("add", &PhraseDataList::add, "Add a sent")
        .def("size", &PhraseDataList::size, "Get size")
        .def("index", &PhraseDataList::index, "Get the sent by index");

    py::class_<EncoderDataList>(m, "EncoderDataList")
        .def(py::init())
        .def("add", &EncoderDataList::add, "Add a sent")
        .def("size", &EncoderDataList::size, "Get size")
        .def("index", &EncoderDataList::index, "Get the sent by index");

    py::class_<ChunkDataList>(m, "ChunkDataList")
        .def(py::init())
        .def("add", &ChunkDataList::add, "Add a sent")
        .def("size", &ChunkDataList::size, "Get size")
        .def("index", &ChunkDataList::index, "Get the sent by index");
}